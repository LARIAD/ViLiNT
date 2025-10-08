import sys
from pathlib import Path

# ROS
from sensor_msgs.msg import Image
import tf_transformations

# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple

deployment_dir = Path(__file__).resolve().parent
project_root = deployment_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# models
sys.path.insert(0, str(project_root / "diffusion"))
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from mnt_train.data.data_utils import IMAGE_ASPECT_RATIO
from mnt_train.training.train_utils import voxelize_pc_for_models

from mnt_train.models.vilint.vilint import Lint_obs, Lint, DenseNetwork, CollisionScoringHeadSeq

def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    model_type = config["model_type"]
    if config["model_type"] == "vilint":
        if config["vision_encoder"] == "vilint":
            vision_encoder = Lint_obs(
                obs_encoding_size=config["encoding_size"],
                im_encoder=config["im_encoder"],
                pc_encoder=config["pc_encoder"],
                context_size=config["context_size"],
                context_size_li=config["context_size_li"],
                use_physics_encoding=config["use_physics_encoding"],
                freeze_pc_encoder=config["freeze_pc_encoder"],
                pc_encoder_channels=config["pc_encoder_channels"],
                physics_dim=config["physics_dim"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
        noise_pred_net = ConditionalUnet1D(
                input_dim=2+2*int(config["learn_angle"]),
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        collision_head = CollisionScoringHeadSeq(
            d_model=config["encoding_size"],
            nheads=config["mha_num_attention_heads"],
            max_steps=config["len_traj_pred"],
            use_length_cond=config["use_length_cond"],
        )
        model = Lint(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
            collision_head=collision_head,
        )

        try:
            if config.get("model_type") == "vilint":
                _print_vilint_parameter_counts(model)
        except Exception as e:
            print(f"[ViLiNT] Parameter count failed: {e}")
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if model_type == "vilint":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    
    return pil_image


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        pil_img = pil_img.convert("RGB")
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)
    

# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi

def process_lidar(lidar_array_list, pc_encoder_channels, device, model_type):
    # Build a dense [CT, M, 3] by per-frame crop->pad/sample
    frames = []
    for pc in lidar_array_list:
        mask = (pc[:,0]>0.0) & (pc[:,0]<20.0) & (pc[:,1]>-20.0) & (pc[:,1]<20.0)
        pc = pc[mask]
        pc = pad_point_cloud(pc, 16384)  # (M,3) numpy
        frames.append(torch.from_numpy(pc).float())
    obs_lidar = torch.stack(frames, dim=0).to(device)  # [CT, M, 3]
    obs_lidar = obs_lidar.unsqueeze(0)  # [1, CT, M, 3]

    return voxelize_pc_for_models(
        obs_lidar=obs_lidar,
        device=device,
        model_type=model_type,
        grid_size=0.02,
        use_grid_coord=True,
        me_quantization_size=0.02,
        me_feat_channels=pc_encoder_channels,
    )

def get_robot_config(config, device):

    robot_config = config.get('robot')
    assert robot_config is not None, "No robot parameters in the chosen dataset."
    lidar_height = robot_config['lidar_height']
    width = robot_config['width']
    height = robot_config['height']
    length = robot_config['length']
    ground_clearance = robot_config['ground_clearance']

    robot_params_tensor = torch.tensor([ width, length, lidar_height], dtype=torch.float32)
    # robot_params_tensor = torch.tensor([lidar_height, width, height, length, ground_clearance], dtype=torch.float32)

    return robot_params_tensor.to(device)

def get_goal_direction(current_position, goal_position, quaternion):
    _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)
    delta_x = goal_position[0] - current_position[0]
    delta_y = goal_position[1] - current_position[1]
    rotated_delta_x = delta_x * np.cos(yaw) + delta_y * np.sin(yaw)
    rotated_delta_y = -delta_x * np.sin(yaw) + delta_y * np.cos(yaw)
    return torch.tensor([rotated_delta_x, rotated_delta_y])

def pad_point_cloud(pc, max_points):
        # If pc is a numpy array, convert it to a torch tensor with the desired dtype.
        num_points = pc.shape[0]
        if isinstance(pc, np.ndarray):
            if num_points < max_points:
                pad_size = max_points - num_points
                pad = np.zeros((pad_size, pc.shape[1]), dtype=pc.dtype)
                return np.vstack([pc, pad])
            else:
                # Generate a random permutation of indices
                perm = np.random.permutation(num_points)[:max_points]
                return pc[perm]
        
        if num_points < max_points:
            pad_size = max_points - num_points
            pad = torch.zeros((pad_size, pc.shape[1]), dtype=pc.dtype).to(pc.device)
            return torch.cat([pc, pad], dim=0)
        else:
            # Generate a random permutation of indices
            perm = torch.randperm(num_points)
            # Select the first max_points indices from the permutation
            sampled_indices = perm[:max_points]
            return pc[sampled_indices]
        
def select_mode(traj: np.ndarray, col: np.ndarray, mode: str = "max", goal_coord: np.ndarray = None):
    """
    Select the trajectory with minimal summarized collision risk.

    Args:
        traj: array of shape (N, T, 2) – N candidate trajectories
        col:  array of shape (N, T) with per-waypoint probs in [0,1], or (N,) already summarized
        mode: 'max' (conservative) or 'mean' to summarize per-trajectory risk when col is (N, T)

    Returns:
        (best_traj: (T, 2), best_risk: float)
    """
    col = np.asarray(col)
    if col.ndim == 2:
        if mode == "max":
            clearance = col.max(axis=1)   # worst waypoint per trajectory
        elif mode == "mean":
            clearance = col.mean(axis=1)  # average risk per trajectory
        elif mode == "min":
            clearance = col.min(axis=1)  # safest waypoint per trajectory
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    elif col.ndim == 1:
        clearance = col
    else:
        raise ValueError(f"Unexpected collision array shape: {col.shape}")

    idxs = (clearance > 3)
    clear = False
    
    if idxs.sum() > 0:
        d_goal = np.linalg.norm(goal_coord - traj[idxs,-1,:2], axis=-1)
        idx = np.argmin(d_goal)
        best_clearance = float(clearance[idxs][idx])
        d_goal = d_goal[idx]
        clear = True
    else:
        idx = np.argmax(clearance)  # if no traj is safe, pick the least bad one
        best_clearance = float(clearance[idx])
        d_goal = np.linalg.norm(goal_coord - traj[idx,-1,:2], axis=-1)
        

    print("Traj selected collision score:", best_clearance)
    return traj[idx], best_clearance, d_goal, clear

# ---- Parameter counting utilities for ViLiNT ----
def _count_params(module: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params) for a torch.nn.Module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def _print_vilint_parameter_counts(model: nn.Module) -> None:
    """Pretty-print parameter counts for Lint / Lint_imle models."""
    # Collect known parts if they exist on the model
    parts = {}
    if hasattr(model, "vision_encoder"):
        parts["vision_encoder"] = model.vision_encoder
    if hasattr(model, "noise_pred_net"):
        parts["noise_pred_net"] = model.noise_pred_net
    if hasattr(model, "policy_net"):
        parts["policy_net"] = model.policy_net
    if hasattr(model, "dist_pred_net"):
        parts["dist_pred_net"] = model.dist_pred_net
    if hasattr(model, "collision_head"):
        parts["collision_head"] = model.collision_head

    total_all, trainable_all = _count_params(model)
    print(
        f"[ViLiNT] Parameters: total={total_all:,} ({total_all/1e6:.2f}M), "
        f"trainable={trainable_all:,} ({trainable_all/1e6:.2f}M)"
    )
    for name, part in parts.items():
        t, tr = _count_params(part)
        print(
            f"[ViLiNT]  - {name:14s}: total={t:,} ({t/1e6:.2f}M), "
            f"trainable={tr:,} ({tr/1e6:.2f}M)"
        )