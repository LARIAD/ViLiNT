import numpy as np
import os
from PIL import Image
from typing import Any, Iterable, Tuple

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import io
from typing import Union, List
import zarr
from pathlib import Path
import re
import tarfile
from typing import Optional

VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training


# --- TAR image helpers -------------------------------------------------------
# Cache mapping tar path -> {index:int -> member_name:str}
_TAR_MEMBER_INDEX_CACHE: dict[str, dict[int, str]] = {}

def _build_tar_member_index(tar_path: str) -> dict[int, str]:
    """Scan a tar and build an index mapping integer frame indices -> member name.
    Accepts names like 12.jpg, 000012.jpeg, 12.PNG, possibly within subfolders.
    """
    idx: dict[int, str] = {}
    try:
        with tarfile.open(tar_path, mode="r") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                base = os.path.basename(m.name)
                m0 = re.match(r"^0*([0-9]+)\.(jpe?g|png)$", base, flags=re.IGNORECASE)
                if not m0:
                    continue
                i = int(m0.group(1))
                # Keep first occurrence if duplicates exist
                if i not in idx:
                    idx[i] = m.name
    except Exception as e:
        print(f"[WARN] Failed to index tar '{tar_path}': {e}")
    return idx

def _get_tar_member_name(tar_path: str, index: int) -> Optional[str]:
    mapping = _TAR_MEMBER_INDEX_CACHE.get(tar_path)
    if mapping is None:
        mapping = _build_tar_member_index(tar_path)
        _TAR_MEMBER_INDEX_CACHE[tar_path] = mapping
    return mapping.get(int(index))

def get_image_bytes_from_images_tar(traj_dir: Union[str, Path], index: int) -> bytes:
    """Return raw bytes of the image with numeric `index` inside `<traj_dir>/images.tar`.
    Supports zero-padded names like 0001.jpeg. Raises FileNotFoundError if missing.
    """
    tar_path = os.path.join(str(traj_dir), "images.tar")
    member_name = _get_tar_member_name(tar_path, int(index))
    if member_name is None:
        # Last-resort fallback: try a few direct candidate names
        candidates = [
            f"{int(index)}.jpg", f"{int(index)}.jpeg",
            f"{int(index):04d}.jpg", f"{int(index):04d}.jpeg",
            f"{int(index):06d}.jpg", f"{int(index):06d}.jpeg",
        ]
        with tarfile.open(tar_path, mode="r") as tf:
            for c in candidates:
                try:
                    ti = tf.getmember(c)
                except KeyError:
                    # Try with a leading './'
                    try:
                        ti = tf.getmember(os.path.join(".", c))
                    except KeyError:
                        ti = None
                if ti is not None and ti.isfile():
                    fobj = tf.extractfile(ti)
                    if fobj is not None:
                        return fobj.read()
        raise FileNotFoundError(f"Image index {index} not found inside {tar_path}")
    with tarfile.open(tar_path, mode="r") as tf:
        fobj = tf.extractfile(member_name)
        if fobj is None:
            raise FileNotFoundError(f"Member '{member_name}' not found in {tar_path}")
        return fobj.read()



def get_collision_status_path(data_folder: str, f: str):
    return os.path.join(data_folder, f, "collision_scores.npy")

def get_width_curve_path(data_folder: str, f: str):
    return os.path.join(data_folder, f, "width_curve.zarr")

def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

def list_to_local_coords(positions: List[float], curr_pos: np.ndarray, curr_yaw: float) -> list[np.ndarray]:
    rotmat = yaw_rotmat(curr_yaw)
    local_pos = []
    for pos in positions:
        if pos.shape[-1] == 2:
            rotmat = rotmat[:2, :2]
        elif pos.shape[-1] == 3:
            pass
        else:
            raise ValueError
        local_pos.append((pos - curr_pos).dot(rotmat))
    return local_pos


def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)


def calculate_deltas(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate deltas between waypoints

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: deltas
    """
    num_params = waypoints.shape[1]
    origin = torch.zeros(1, num_params)
    prev_waypoints = torch.concat((origin, waypoints[:-1]), axis=0)
    deltas = waypoints - prev_waypoints
    if num_params > 2:
        return calculate_sin_cos(deltas)
    return deltas


def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate sin and cos of the angle

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: waypoints with sin and cos of the angle
    """
    assert waypoints.shape[1] == 3
    angle_repr = torch.zeros_like(waypoints[:, :2])
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


def transform_images(
    img: Image.Image, transform: transforms, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    viz_img = img.resize(VISUALIZATION_IMAGE_SIZE)
    viz_img = TF.to_tensor(viz_img)
    img = img.resize(image_resize_size)
    transf_img = transform(img)
    return viz_img, transf_img


def resize_and_aspect_crop(
    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img


def img_path_to_data(buf: io.BytesIO, image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """
    Load an image from an in-memory buffer (io.BytesIO or file-like object) and transform it.
    Args:
        buf (io.BytesIO): in-memory image buffer or file-like object
        image_resize_size (Tuple[int, int]): size to resize the image to
    Returns:
        torch.Tensor: resized image as tensor
    """
    return resize_and_aspect_crop(Image.open(buf), image_resize_size)

def lidar_path_to_data(source: Union[str, Path, io.BytesIO, np.ndarray, torch.Tensor],
                       point_cloud_size: int,
                       time: Optional[int] = None) -> torch.Tensor:
    """Load / format a point cloud.
    - If `source` is a path to a Zarr directory, you must pass `time` and it will read points[time].
    - If `source` is a numpy/torch array of shape (P,3), it will be cropped/padded.
    Returns a (point_cloud_size, 3) torch.FloatTensor.
    """
    try:
        if isinstance(source, (str, Path)):
            # Path to a Zarr array directory
            if time is None:
                raise ValueError("When 'source' is a path, 'time' must be provided.")
            pc_arr = _open_zarr_array(Path(source))
            point_cloud = pc_arr[int(time)].astype(np.float32, copy=False)
        elif isinstance(source, io.BytesIO):
            # Not supporting npy-bytes here; prefer array or zarr path
            raise TypeError("BytesIO source not supported for point clouds; pass array or zarr path.")
        else:
            # Array-like
            point_cloud = np.asarray(source, dtype=np.float32)
        # Simple ROI crop then pad/sample
        if point_cloud.ndim != 2 or point_cloud.shape[1] < 2:
            return torch.zeros((point_cloud_size, 3), dtype=torch.float32)
        mask = (
            (point_cloud[:, 0] > 0.0) & (point_cloud[:, 0] < 20.0) &
            (point_cloud[:, 1] < 20.0) & (point_cloud[:, 1] > -20.0)
        )
        point_cloud = point_cloud[mask]
        if point_cloud.shape[1] == 2:
            # add z=0 if absent
            point_cloud = np.concatenate([point_cloud, np.zeros((point_cloud.shape[0], 1), dtype=point_cloud.dtype)], axis=1)
        point_cloud_torch = torch.from_numpy(point_cloud.copy()).float()
        quantized_pc = pad_point_cloud(point_cloud_torch, point_cloud_size)
        return quantized_pc
    except Exception as e:
        print(f"Failed to load point cloud from {source}: {e}")
        return torch.zeros((point_cloud_size, 3), dtype=torch.float32).contiguous()
    
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
            pad = torch.zeros((pad_size, pc.shape[1]), dtype=pc.dtype)
            return torch.cat([pc, pad], dim=0)
        else:
            # Generate a random permutation of indices
            perm = torch.randperm(num_points)
            # Select the first max_points indices from the permutation
            sampled_indices = perm[:max_points]
            return pc[sampled_indices]

def _open_zarr_array(path_dir: Path):
    store = zarr.storage.LocalStore(str(path_dir))
    return zarr.open_array(store, mode="r")
