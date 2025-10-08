
import os
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import tqdm
import itertools

from typing import List, Optional, Dict

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from mnt_train.training.logger import Logger
from diffusers.training_utils import EMAModel
from mnt_train.data.data_utils import VISUALIZATION_IMAGE_SIZE

from mnt_train.visualizing.action_utils import plot_trajs_and_points

# LOAD DATA CONFIG
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 4)
    ndeltas = to_numpy(ndeltas)
    ndeltas[...,:2] = unnormalize_data(ndeltas[...,:2], action_stats)
    actions = np.cumsum(ndeltas[...,:2], axis=1)
    actions = np.concatenate((actions, ndeltas[...,2:]), axis=-1)
    return from_numpy(actions).to(device)

def voxelize_pc_for_models(
    obs_lidar: torch.Tensor,
    device: torch.device,
    *,
    model_type: str,
    lidar_mask : Optional[torch.Tensor] = None,
    # PointTransformer
    grid_size: float = 0.02,
    use_grid_coord: bool = True,
    # Minkowski / MinkUNet
    me_quantization_size: float = 0.02,
    me_feat_channels: int = 3,  # 3 -> xyz features, 1 -> ones
):
    B, CT, N, D = obs_lidar.shape
    assert D == 3

    pcs = obs_lidar.to(device=device, dtype=torch.float32).reshape(B * CT, N, 3)
    mt = model_type.lower()

    with torch.no_grad():
        cloud_min = pcs.amin(dim=1)          # (BCT, 3)
        cloud_max = pcs.amax(dim=1)          # (BCT, 3)
        deg = (cloud_max - cloud_min).abs().sum(dim=1) == 0  # (BCT,)
        if deg.any():
            idx = torch.nonzero(deg, as_tuple=False).squeeze(1)
            # N>=2: move one point by one voxel to create non-zero extent
            if pcs.shape[1] >= 2:
                pcs[idx, 1, 0] = pcs[idx, 1, 0] + float(grid_size)
            else:
                pcs[idx, 0, 0] = pcs[idx, 0, 0] + float(grid_size)


    # -------- PointTransformer output --------
    if mt == "pointtransformerv3":
        coord = pcs.reshape(B * CT * N, 3)
        batch = torch.arange(B * CT, device=device, dtype=torch.long).repeat_interleave(N)
        feat = torch.cat([coord, torch.zeros((coord.shape[0], 1), device=device)], dim=1)

        out = {
            "coord": coord,
            "batch": batch,
            "feat": feat,  # simplest: xyz as features
            "grid_size": torch.tensor(grid_size, device=device),
        }

        if use_grid_coord:
            out["grid_coord"] = torch.div(
                coord - coord.min(dim=0).values,
                grid_size,
                rounding_mode="trunc",
            ).to(torch.int32)

        return out

    # -------- MinkowskiEngine output --------
    if mt == "minkunet":
        import MinkowskiEngine as ME

        coords_list, idx_list = zip(*(
            ME.utils.sparse_quantize(pc, quantization_size=me_quantization_size, return_index=True)
            for pc in pcs
        ))
        coords = ME.utils.batched_coordinates(list(coords_list)).to(device)

        if me_feat_channels == 3:
            features = torch.cat([pc[idx] for pc, idx in zip(pcs, idx_list)], dim=0).to(device).float()
        else:
            features = torch.ones((coords.shape[0], 1), device=device, dtype=torch.float32)

        return {"coords": coords, "features": features}

    raise ValueError(f"Unknown model_type={model_type!r}. Expected 'pointtransformerv3' or 'minkunet'.")

def asymmetric_huber_loss(logits, targets, w_over):
    e = logits - targets
    w = torch.where(e > 0, w_over, 1.0)
    base = F.smooth_l1_loss(logits, targets, reduction='none', beta=1.0)
    return w * base
