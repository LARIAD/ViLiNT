"""
Generate clearance-based supervision targets from per-trajectory LiDAR folders.

Expected input per trajectory:
  - one `.pkl` file containing `position` and `yaw`
  - a sequence of point clouds stored as numbered `.npy` files

For each trajectory, this script writes:
  - `clearance_widths.npy`
  - `width_curve.npy`
  - optional TTC outputs (`ttc_table.npy`, `ttc_widths.npy`)
  - preprocessed point clouds as `<frame>_ng.npy`
"""

import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mnt_train.data.data_utils import to_local_coords
import time

from LiDARtoolkit import removeGround

from typing import Optional

DEBUG = False
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Vectorized distance helpers ---
def _point_to_segment_distance(points: np.ndarray,
                               seg_start: np.ndarray,
                               seg_end: np.ndarray) -> np.ndarray:
    """Vectorized distance from many 2D points to many line segments.
    points:   [M,2]
    seg_start:[L,2]
    seg_end:  [L,2]
    returns:  [M,L] distances from each point to each segment.
    """
    if points.size == 0 or seg_start.shape[0] == 0:
        return np.zeros((points.shape[0], seg_start.shape[0]), dtype=np.float32)
    P = points.astype(np.float32, copy=False)
    A = seg_start.astype(np.float32, copy=False)
    B = seg_end.astype(np.float32, copy=False)

    AB = B - A                       # [L,2]
    AP = P[:, None, :] - A[None, :, :]  # [M,L,2]

    # segment parameter t in [0,1]
    AB2 = np.maximum(np.sum(AB * AB, axis=1), 1e-12)   # [L]
    t = np.clip(np.sum(AP * AB, axis=-1) / AB2[None, :], 0.0, 1.0)  # [M,L]

    proj = A[None, :, :] + t[..., None] * AB[None, :, :]            # [M,L,2]
    diff = P[:, None, :] - proj                                      # [M,L,2]
    d = np.linalg.norm(diff, axis=-1).astype(np.float32)             # [M,L]
    return d

def _point_to_normal_distance(points: np.ndarray,
                              seg_start: np.ndarray,
                              seg_end: np.ndarray) -> np.ndarray:
    """Vectorized distance from many 2D points to many line segments normals.
    points:   [M,2]
    seg_start:[L,2]
    seg_end:  [L,2]
    returns:  [M,L] distances from each point to each segment.
    """
    if points.size == 0 or seg_start.shape[0] == 0:
        return np.zeros((points.shape[0], seg_start.shape[0]), dtype=np.float32)
    P = points.astype(np.float32, copy=False)
    A = seg_start.astype(np.float32, copy=False)
    B = seg_end.astype(np.float32, copy=False)

    AB = B - A                       # [L,2]
    ABn = np.stack((-AB[..., 1], AB[..., 0]), axis=1)
    AP = P[:, None, :] - A[None, :, :]  # [M,L,2]

    # segment parameter t in [0,1]
    ABn2 = np.maximum(np.sum(ABn * ABn, axis=1), 1e-12)   # [L]
    t = np.clip(np.sum(AP * ABn, axis=-1) / ABn2[None, :], 0.0, 1.0)  # [M,L]

    proj = A[None, :, :] + t[..., None] * ABn[None, :, :]            # [M,L,2]
    diff = P[:, None, :] - proj                                      # [M,L,2]
    d = np.linalg.norm(diff, axis=-1).astype(np.float32)             # [M,L]
    return d


def _polyline_prefix_min_widths(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Compute per-lookahead critical widths for a polyline in one pass.
    Returns width_curve[j-1] = 2 * min_{p} dist(p, polyline[0..j]).
    points:   [M,2]
    polyline: [K,2] (K>=2)
    returns:  [K-1] width curve (float32)
    """
    if polyline.shape[0] < 2 or points.size == 0:
        return np.zeros((0,), dtype=np.float32)
    A = polyline[:-1]
    B = polyline[1:]
    d_mat = _point_to_segment_distance(points, A, B)   # [M,L]
    d_len = _point_to_normal_distance(points, A, B)    # [M,L]
    seg_min = np.min(d_mat, axis=0)                    # [L]
    len_min = np.min(d_len, axis=0)                    # [L]
    return (2.0 * seg_min.astype(np.float32)), (2.0 * len_min.astype(np.float32))       # [L], [L]

def _ttc_from_width_curve(width_curve, widths_eval, dt):
    """
    For each width in widths_eval, return the earliest index j where width_curve[j] < width,
    and compute TTC as j * dt. If never violated, return np.inf.
    """
    ttc = np.full_like(widths_eval, np.inf, dtype=np.float32)
    for widx, w in enumerate(widths_eval):
        below = np.where(width_curve < w)[0]
        if below.size > 0:
            ttc[widx] = (below[0] + 1) * dt
    return ttc

def process_trajectory(traj_folder,
                       physics,
                       dt: float = 1.0,
                       horizon: int = 10,
                       forward_range: float = 20.0,
                       lateral_range: float = 20.0,
                       widths_eval: Optional[np.ndarray] = None,
                       flat_pc: bool = True):
    """
    Compute per-timestep *clearance-based* self-supervision targets.

    For each timestep i, we:
      1) Build the polyline of the next `horizon` future positions expressed
         in the robot frame at time i (prepend the origin [0,0]).
      2) From the current LiDAR pc (robot frame), keep points within a window
         x in [0, forward_range], |y| <= lateral_range, and remove ground.
      3) Compute the *critical width curve* over lookaheads j=1..J:
         width_curve[j-1] = 2 * min_{p in pc} dist(p, polyline_prefix_0..j).
      4) Optionally compute a TTC table for a list of candidate widths.

    Returns:
      dict with keys:
        - 'clearance_widths': [N] minimal corridor width (meters) per timestep
        - 'width_curve':      [N, horizon] per-lookahead critical widths (NaN padded)
        - 'ttc_table' (opt):  [N, W] TTC in seconds for each requested width
        - 'ttc_widths' (opt): [W] the widths used
    """

    # Gather trajectory .pkl and LiDAR .npy
    lidar_list = []
    trajectory = None
    for root, dirs, files in os.walk(traj_folder):
        # load the sole .pkl once
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "rb") as f:
                        trajectory = pickle.load(f)
                except Exception as e:
                    print("Error processing file:", file_path, "->", e)
                    continue
        # collect LiDAR frames (ignore any supervision files)
        npy_files = [
            f for f in files
            if f.endswith(".npy")
            and f != "collision_scores.npy"
            and f != "clearance_widths.npy"
            and f != "width_curve.npy"
            and not f.startswith("ttc_table")
            and not f.endswith("_ng.npy")
        ]
        npy_files.sort(key=lambda fn: int(os.path.splitext(fn)[0]))
        for fname in npy_files:
            file_path = os.path.join(root, fname)
            try:
                lidar = np.load(file_path)
                if (lidar.ndim != 2 or lidar.shape[1] != 3 or lidar.shape[0] == 0) and not flat_pc:
                    proc_pc = np.zeros((0, 3), dtype=np.float32)
                elif flat_pc:
                    pc_tmp = lidar.astype(np.float32, copy=False)
                    # ROI in robot frame: x in (0, forward_range], |y| <= lateral_range
                    roi_mask = (pc_tmp[:, 0] > 0.0) & (pc_tmp[:, 0] <= forward_range) & (np.abs(pc_tmp[:, 1]) <= lateral_range)
                    pc_tmp = pc_tmp[roi_mask]
                    proc_pc = pc_tmp if pc_tmp.ndim == 2 and pc_tmp.shape[1] == 3 else np.zeros((0, 3), dtype=np.float32)
                else:
                    pc_tmp = lidar.astype(np.float32, copy=False)
                    # ROI in robot frame: x in (0, forward_range], |y| <= lateral_range
                    roi_mask = (pc_tmp[:, 0] > 0.0) & (pc_tmp[:, 0] <= forward_range) & (np.abs(pc_tmp[:, 1]) <= lateral_range)
                    pc_tmp = pc_tmp[roi_mask]
                    try:
                        pc_tmp = removeGround(pc_tmp).to_numpy().astype(np.float32)
                    except Exception:
                        pc_tmp = np.asarray(pc_tmp, dtype=np.float32)
                    proc_pc = pc_tmp if pc_tmp.ndim == 2 and pc_tmp.shape[1] == 3 else np.zeros((0, 3), dtype=np.float32)

                # Save preprocessed point cloud next to the original as `<stem>_ng.npy`
                stem = os.path.splitext(fname)[0]
                out_ng_path = os.path.join(root, f"{stem}_ng.npy")
                try:
                    np.save(out_ng_path, proc_pc)
                except Exception as e:
                    print("Error saving preprocessed point cloud:", out_ng_path, "->", e)

                # Append preprocessed pc to the working list
                lidar_list.append(proc_pc)
                # Continue to next file
                continue
            except Exception as e:
                print("Error processing file:", file_path, "->", e)
                continue

    # sanity checks
    if trajectory is None:
        print(f"No .pkl file in {traj_folder}")
        return None

    try:
        positions = np.asarray(trajectory['position'], dtype=np.float32)  # [N,3]
        yaws      = np.asarray(trajectory['yaw'], dtype=np.float32)       # [N]
    except Exception as e:
        print(f"Malformed trajectory in {traj_folder}: {e}")
        return None

    if len(lidar_list) == 0:
        print("No lidar point clouds")
        return None

    N = min(len(positions), len(lidar_list))
    if N < 2:
        print("Not enough steps to compute future polyline")
        return None

    clearance_widths = np.zeros((N,), dtype=np.float32)
    width_curve_table = np.full((N, horizon, 2), np.nan, dtype=np.float32)

    ttc_table = None
    if widths_eval is not None and len(widths_eval) > 0:
        widths_eval = np.array(widths_eval, dtype=np.float32)
        ttc_table = np.full((N, widths_eval.shape[0]), np.inf, dtype=np.float32)

    for i in range(N):
        start = time.time()
        # future indices within horizon
        j_end = min(i + horizon, N - 1)
        future_world = positions[i+1:j_end+1, :2]  # [J,2] next positions in world
        yaw_i = float(yaws[i])

        # polyline in robot frame at time i (prepend origin)
        if future_world.shape[0] == 0:
            poly_local = np.zeros((1, 2), dtype=np.float32)
        else:
            poly_local = to_local_coords(future_world, positions[i, :2], yaw_i).astype(np.float32)
            poly_local = np.vstack([np.zeros((1, 2), dtype=np.float32), poly_local])  # [J+1,2]

        # Use preprocessed (ROI-filtered + ground-removed) point cloud loaded earlier
        pc = lidar_list[i].astype(np.float32, copy=False)
        if pc.size == 0:
            clearance_widths[i] = 2.0 * lateral_range
            if ttc_table is not None:
                ttc_table[i, :] = np.inf
            continue
        pc_xy = pc[:, :2]
        if pc_xy.size == 0 or poly_local.shape[0] < 2:
            clearance_widths[i] = 2.0 * lateral_range
            if ttc_table is not None:
                ttc_table[i, :] = np.inf
            continue

        # compute width curve over prefixes 1..J
        width_curve, len_curve = _polyline_prefix_min_widths(pc_xy, poly_local)

        # store per-lookahead critical widths (NaN-padded to horizon)
        if width_curve.size > 0:
            width_curve_table[i, :width_curve.shape[0], 0] = width_curve
        if len_curve.size > 0:
            width_curve_table[i, :len_curve.shape[0], 1] = len_curve

        # best (smallest) width along the horizon
        clearance_widths[i] = float(np.min(width_curve)) if width_curve.size > 0 else 2.0 * lateral_range

        # TTC table if requested
        if ttc_table is not None:
            ttc_table[i] = _ttc_from_width_curve(width_curve, widths_eval, dt)

        if DEBUG:
            print(f"Time to process point: {time.time() - start}")

    result = {
        'clearance_widths': clearance_widths,
        'width_curve': width_curve_table,
    }
    if ttc_table is not None:
        result['ttc_table'] = ttc_table
        result['ttc_widths'] = widths_eval

    print(f"[OK] clearance width vector shape: {clearance_widths.shape}")
    print(f"[OK] width curve table shape: {width_curve_table.shape}")
    if ttc_table is not None:
        print(f"[OK] TTC table shape: {ttc_table.shape} (widths: {list(widths_eval)})")

    return result



def store_collision_trajs_yaml(parent_folder: str,
                               output_yaml: str = 'collision_trajs.yml'):
    """
    Legacy helper that scans each subfolder of parent_folder for
    `collision_matrix.csv` and stores matching trajectory names in a YAML file.

    :param parent_folder: path containing trajectory subfolders
    :param output_yaml:  path to write the YAML output
    """
    collision_trajs = []

    # iterate over trajectory subfolders
    for name in sorted(os.listdir(parent_folder)):
        traj_dir = os.path.join(parent_folder, name)
        csv_path = os.path.join(traj_dir, 'collision_matrix.csv')
        if not os.path.isfile(csv_path):
            continue

        # load and check for any collision flags (1)
        df = pd.read_csv(csv_path)
        # assume first column is 'index', so drop it before checking
        flags = df.drop(columns=[df.columns[0]])
        if (flags.values is not None):
            collision_trajs.append(name)
    p = Path(parent_folder).resolve()
    name = p.parent.name

    if os.path.exists(output_yaml):
        try:
            with open(output_yaml, "r") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            print("Error reading the YAML file:", exc)
            return
    else:
        data = {}

    data[name] = collision_trajs

    # write out to YAML
    with open(output_yaml, 'w') as f:
        yaml.safe_dump(data, f)

    print(f"→ Found {len(collision_trajs)} trajectories with collisions.")
    print(f"→ YAML written to {output_yaml}")

def get_data_config(data_config, dataset):
    robot_config = data_config[dataset].get('robot')
    width = robot_config['width']
    height = robot_config['height']
    length = robot_config['length']
    ground_clearance = robot_config['ground_clearance']
    max_speed = robot_config['max_speed']
    max_yaw_rate = robot_config['max_yaw_rate']
    max_acceleration = robot_config['max_acceleration']
    robot_params_tensor = np.array([width, length, max_speed, max_yaw_rate, max_acceleration])

    return robot_params_tensor

def main(parent_folders, dataset):
    with open(
            os.path.join(SCRIPT_DIR, "../data/data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
    assert (
        dataset in all_data_config
    ), f"Dataset {dataset} not found in data_config.yaml"
    try:
        physics = get_data_config(all_data_config, dataset)
    except Exception as e:
        physics = None
    for i, parent_folder in enumerate(parent_folders):
        # List only directories (ignore files) in parent_folder
        subdirs = [
            os.path.join(parent_folder, d) for d in os.listdir(parent_folder)
            if os.path.isdir(os.path.join(parent_folder, d))
        ]
        if not subdirs:
            print(f"No subfolders found in {parent_folder}.")
            return

        for traj in sorted(subdirs):
            print(f"Processing {traj} ...")
            res = process_trajectory(traj, physics, dt=1.0, horizon=args.horizon, widths_eval=np.array(args.widths, dtype=np.float32), flat_pc=args.flat)
            if res is None:
                continue
            np.save(os.path.join(traj, "clearance_widths.npy"), res['clearance_widths'])
            np.save(os.path.join(traj, "width_curve.npy"), res['width_curve'])
            if 'ttc_table' in res:
                np.save(os.path.join(traj, "ttc_table.npy"), res['ttc_table'])
                np.save(os.path.join(traj, "ttc_widths.npy"), res['ttc_widths'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Batch-process multiple trajectories for collision flags"
    )
    parser.add_argument('parent_folders', nargs='+',
                        help="Directories containing trajectory subfolders")
    parser.add_argument("--dataset", "-d", type=str)
    parser.add_argument("--horizon", type=int, default=10, help="Lookahead horizon for width curve")
    parser.add_argument("--widths", nargs='*', type=float, default=[], help="Widths for TTC computation")
    parser.add_argument("--flat", action='store_true', help="Use 2D point clouds (ignore z-axis)")
    args = parser.parse_args()

    main(args.parent_folders, args.dataset)
