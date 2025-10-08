#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_width_corridors.py

Visualize corridor "width_curve" around the future polyline in the robot frame,
overlaid on preprocessed LiDAR frames.

Two input modes:
  (A) Legacy folders (use with --use-tar):
      - one .pkl with keys: 'position' (Nx3), 'yaw' (N,)
      - LiDAR frames: <index>.npy and <index>_ng.npy
      - width_curve.npy  (shape [N,H,2] or [N,2,H])
  (B) build_dataset-style folders (default):
      - pose_pos.zarr, pose_orien.zarr  (preferred)
        *if missing*, the script will fall back to a `.pkl` trajectory with
        keys such as 'position'/'positions'/'pose'/'pos' and 'yaw'/'yaws'/'theta'/'heading'
      - points_ng.zarr (+ points_ng_lengths.zarr)  [preferred if present],
        otherwise points.zarr
      - width_curve.zarr (preferred) or width_curve.npy

Usage examples:
  # Legacy mode (per-frame npy)
  python visualize_width_corridors.py /path/to/trajectory --use-tar \
      --out ./corridor_viz --max-frames 200 --step 1 --z-max 30

  # Zarr mode (build_dataset outputs in a single folder)
  python visualize_width_corridors.py /path/to/trajectory  \
      --out ./corridor_viz --max-frames 200 --step 1 --z-max 30
"""

import os
import re
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import FuncNorm
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from pathlib import Path
import zarr

# -------------------------
# Helpers
# -------------------------

# --- Zarr helpers & orientation utilities ---

def _open_zarr_array(path_dir: Path):
    store = zarr.storage.DirectoryStore(str(path_dir))
    return zarr.open_array(store, mode="r")

def _quat_to_yaw_xyzw(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))

def _extract_yaw_from_orientations(ori: np.ndarray) -> np.ndarray:
    ori = np.asarray(ori)
    if ori.ndim != 2:
        raise ValueError(f"orientation should be 2-D, got shape {ori.shape}")
    N, D = ori.shape
    if D == 4:  # quaternion (x,y,z,w)
        x, y, z, w = ori.T
        return np.array([_quat_to_yaw_xyzw(x[i], y[i], z[i], w[i]) for i in range(N)], dtype=np.float32)
    if D == 3:  # Euler (roll, pitch, yaw)
        return ori[:, 2].astype(np.float32, copy=False)
    if D >= 7:  # take last 4 as quaternion
        x, y, z, w = ori[:, -4:].T
        return np.array([_quat_to_yaw_xyzw(x[i], y[i], z[i], w[i]) for i in range(N)], dtype=np.float32)
    raise ValueError(f"Unsupported orientation shape (N,{D}). Expected 3 (RPY) or 4 (xyzw quaternion).")

def _load_zarr_bundle(traj_dir: str):
    """
    Load build_dataset-style trajectory bundle from `traj_dir`.
    Returns dict with keys: pos (Nx3), yaw (N,), width_curve (N,H,2), get_pc (callable(i)->(M,3)).
    Prefers width_curve.zarr and points_ng.zarr if available.
    If pose_pos.zarr / pose_orien.zarr are missing, falls back to a `.pkl` trajectory
    (keys like 'position'/'positions'/'pose'/'pos' and 'yaw'/'yaws'/'theta'/'heading').
    """
    tdir = Path(traj_dir)
    pos_p = tdir / "pose_pos.zarr"
    ori_p = tdir / "pose_orien.zarr"
    wc_z = tdir / "width_curve.zarr"
    wc_npy = tdir / "width_curve.npy"
    pts_ng_p = tdir / "points_ng.zarr"
    pts_ng_len_p = tdir / "points_ng_lengths.zarr"
    pts_p = tdir / "points.zarr"

    # 1) Load positions and yaw: prefer zarr, else fallback to a trajectory .pkl
    pos = None
    yaw = None
    if pos_p.exists() and ori_p.exists():
        pos = _open_zarr_array(pos_p)[:].astype(np.float32, copy=False)
        if pos.ndim != 2 or pos.shape[1] < 2:
            raise ValueError(f"pose_pos.zarr must have shape (N,>=2), got {pos.shape}")
        if pos.shape[1] > 3:
            pos = pos[:, :3]
        ori = _open_zarr_array(ori_p)[:].astype(np.float32, copy=False)
        yaw = _extract_yaw_from_orientations(ori)
    else:
        # Fallback: find a .pkl trajectory like in process_lidar_col_zarr.py
        pkl_path = None
        for p in tdir.rglob('*.pkl'):
            pkl_path = p
            break
        if pkl_path is None:
            raise FileNotFoundError(
                f"Expected pose_pos.zarr/pose_orien.zarr or a trajectory .pkl in {traj_dir}")
        with open(pkl_path, 'rb') as f:
            traj = pickle.load(f)
        # Robust position key
        pos_key = None
        for k in ('position', 'positions', 'pose', 'pos'):
            if isinstance(traj, dict) and k in traj:
                pos_key = k
                break
        if pos_key is None:
            raise KeyError(".pkl trajectory missing a positions array (expected one of: 'position', 'positions', 'pose', 'pos')")
        pos = np.asarray(traj[pos_key], dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] < 2:
            raise ValueError(f"Trajectory positions must have shape (N,>=2), got {pos.shape}")
        if pos.shape[1] > 3:
            pos = pos[:, :3]
        # Robust yaw key
        yaw_key = None
        for k in ('yaw', 'yaws', 'theta', 'heading'):
            if isinstance(traj, dict) and k in traj:
                yaw_key = k
                break
        if yaw_key is None:
            raise KeyError(".pkl trajectory missing a yaw array (expected one of: 'yaw', 'yaws', 'theta', 'heading')")
        yaw = np.asarray(traj[yaw_key], dtype=np.float32)

    # 2) width_curve: prefer zarr, else npy. Normalize to (N,H,2)
    if wc_z.exists():
        wc = _open_zarr_array(wc_z)[:].astype(np.float32, copy=False)
    elif wc_npy.exists():
        wc_raw = np.load(wc_npy)
        if wc_raw.ndim != 3:
            raise ValueError(f"width_curve.npy rank {wc_raw.ndim} invalid; shape={wc_raw.shape}")
        if wc_raw.shape[-1] == 2 and wc_raw.shape[1] != 2:
            wc = wc_raw.astype(np.float32, copy=False)
        elif wc_raw.shape[1] == 2:
            wc = np.transpose(wc_raw, (0, 2, 1)).astype(np.float32, copy=False)
        else:
            raise ValueError(f"width_curve.npy shape {wc_raw.shape} invalid; expected (N,H,2) or (N,2,H)")
    else:
        raise FileNotFoundError(f"Neither width_curve.zarr nor width_curve.npy found in {traj_dir}")

    # 3) points provider: prefer points_ng.zarr, else points.zarr
    if pts_ng_p.exists():
        pts_ng = _open_zarr_array(pts_ng_p)
        if pts_ng_len_p.exists():
            lens = _open_zarr_array(pts_ng_len_p)[:].astype(np.int32, copy=False)
        else:
            lens = None
        def _get_pc(i: int):
            pc = np.asarray(pts_ng[int(i)]).astype(np.float32, copy=False)
            if pc.ndim != 2 or pc.shape[1] != 3:
                if pc.ndim == 2 and pc.shape[0] == 3:
                    pc = pc.T
                elif pc.ndim == 3 and pc.shape[-1] == 3:
                    pc = pc.reshape(-1, 3)
            if lens is not None and i < len(lens):
                pc = pc[: int(lens[i])]
            return pc
        get_pc = _get_pc
    elif pts_p.exists():
        pts = _open_zarr_array(pts_p)
        def _get_pc(i: int):
            arr = np.asarray(pts[int(i)])
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr.astype(np.float32, copy=False)
            if arr.ndim == 2 and arr.shape[0] == 3:
                return arr.T.astype(np.float32, copy=False)
            if arr.ndim == 3:
                if arr.shape[-1] == 3:
                    return arr.reshape(-1, 3).astype(np.float32, copy=False)
                if arr.shape[0] == 3:
                    return np.transpose(arr, (1, 2, 0)).reshape(-1, 3).astype(np.float32, copy=False)
            raise ValueError(f"Unsupported points.zarr slice shape {arr.shape}")
        get_pc = _get_pc
    else:
        raise FileNotFoundError(f"Neither points_ng.zarr nor points.zarr found in {traj_dir}")

    # Normalize width_curve to (N,H,2)
    if wc.shape[1] == 2 and wc.shape[-1] != 2:
        wc = np.transpose(wc, (0, 2, 1))

    return {
        "pos": pos,
        "yaw": yaw,
        "width_curve": wc.astype(np.float32, copy=False),
        "get_pc": get_pc,
    }

_num_re = re.compile(r"(\d+)(?:_ng)?\.npy$")

def point_to_segment_distance(points, A, B):
    AB = B - A                     # [L,2]
    AP = points[:, None, :] - A[None, :, :]  # [M,L,2]
    AB2 = np.maximum((AB**2).sum(1), 1e-12)
    t = np.clip((AP * AB).sum(-1) / AB2[None, :], 0.0, 1.0)  # [M,L]
    proj = A[None, :, :] + t[..., None] * AB[None, :, :]     # [M,L,2]
    d = np.linalg.norm(points[:, None, :] - proj, axis=-1)   # [M,L]
    return d, t

def _numeric_stem(path: str) -> int:
    """Extract the numeric stem from '123_ng.npy' or '123.npy'."""
    m = _num_re.search(os.path.basename(path))
    return int(m.group(1)) if m else -1

def to_local_coords(points_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    """
    Transform world XY to robot frame at pose (origin_xy, yaw).
    Robot x-axis points forward; this matches process_lidar_collision.py semantics.
    """
    d = points_xy - origin_xy[None, :]
    c, s = np.cos(yaw), np.sin(yaw)
    R_T = np.array([[ c,  s],
                    [-s,  c]], dtype=np.float32)
    return (d @ R_T.T).astype(np.float32)

def segment_rect_polygon(p0: np.ndarray, p1: np.ndarray, width: float) -> np.ndarray:
    """
    Oriented rectangle polygon for a segment from p0 to p1, with total width 'width'.
    Returns 4x2 vertices in order.
    """
    v = p1 - p0
    L = np.linalg.norm(v)
    if L <= 1e-9 or width <= 0.0:
        half_w = max(width, 1e-6) * 0.5
        return np.array([
            p0 + np.array([-half_w, -half_w]),
            p0 + np.array([ half_w, -half_w]),
            p0 + np.array([ half_w,  half_w]),
            p0 + np.array([-half_w,  half_w]),
        ], dtype=np.float32)

    u = v / L
    n = np.array([-u[1], u[0]], dtype=np.float32)  # left normal
    hw = width * 0.5
    return np.stack([
        p0 - hw * n,
        p0 + hw * n,
        p1 + hw * n,
        p1 - hw * n
    ], axis=0).astype(np.float32)

# -------------------------
# Core visualization
# -------------------------

def visualize_traj_with_corridors(traj_dir: str,
                                  out_dir: str,
                                  max_frames: int = -1,
                                  step: int = 1,
                                  z_clip: float = None,
                                  pc_point_size: float = 5.0,
                                  corridor_alpha: float = 0.15,
                                  use_tar: bool = False):
    # 1) Load trajectory, width_curve, and a point-cloud provider
    if not use_tar:
        # ZARR MODE (default)
        bundle = _load_zarr_bundle(traj_dir)
        pos = bundle["pos"]                  # [N,3]
        yaw = bundle["yaw"]                  # [N]
        width_curve = bundle["width_curve"]  # [N,H,2]
        get_pc = bundle["get_pc"]            # callable(i)->(M,3)
        N_z, H = width_curve.shape[0], width_curve.shape[1]
        N_use = N_z
    else:
        # LEGACY MODE (per-frame *_ng.npy + width_curve.npy + .pkl)
        pkl_path = None
        for f in os.listdir(traj_dir):
            if f.endswith(".pkl"):
                pkl_path = os.path.join(traj_dir, f)
                break
        if pkl_path is None:
            raise FileNotFoundError("No .pkl trajectory file found in: " + traj_dir)

        with open(pkl_path, "rb") as f:
            traj = pickle.load(f)
        pos = np.asarray(traj["position"], dtype=np.float32)  # [N,3]
        yaw = np.asarray(traj["yaw"], dtype=np.float32)       # [N]

        wc_path = os.path.join(traj_dir, "width_curve.npy")
        if not os.path.isfile(wc_path):
            raise FileNotFoundError("width_curve.npy not found in: " + traj_dir)
        wc_raw = np.load(wc_path)
        if wc_raw.ndim != 3:
            raise ValueError(
                f"width_curve.npy has unexpected rank {wc_raw.ndim}; expected a 3D array, got shape {wc_raw.shape}")
        if wc_raw.shape[-1] == 2 and wc_raw.shape[1] != 2:
            width_curve = wc_raw.astype(np.float32, copy=False)
        elif wc_raw.shape[1] == 2:
            width_curve = np.transpose(wc_raw, (0, 2, 1)).astype(np.float32, copy=False)
        else:
            raise ValueError(
                f"width_curve.npy has unexpected shape {wc_raw.shape}; expected either (N, H, 2) or (N, 2, H)")
        N_z, H = width_curve.shape[0], width_curve.shape[1]

        ng_paths = [os.path.join(traj_dir, f) for f in os.listdir(traj_dir) if f.endswith("_ng.npy")]
        if len(ng_paths) == 0:
            raise FileNotFoundError("No *_ng.npy files found in: " + traj_dir)
        ng_paths.sort(key=_numeric_stem)

        def get_pc(i: int):
            return np.load(ng_paths[i]).astype(np.float32, copy=False)

        N_use = min(len(ng_paths), len(pos), len(yaw), N_z)

    # limit number of frames if requested
    if max_frames > 0:
        N_use = min(N_use, max_frames)

    os.makedirs(out_dir, exist_ok=True)

    # 2) Iterate frames
    for i in range(0, N_use, step):
        pc = get_pc(i)
        if pc.ndim != 2 or pc.shape[-1] < 2:
            # best effort coercion
            pc = pc.reshape(-1, pc.shape[-1] if pc.ndim > 1 else 2)
        pc = pc.astype(np.float32, copy=False)

        # Optional Z clip (for nicer color range)
        if z_clip is not None and pc.shape[1] >= 3:
            pc = pc[np.abs(pc[:, 2]) <= float(z_clip)]

        # Rebuild local polyline for the next H steps
        j_end = min(i + H, N_use - 1)
        future_world = pos[i+1:j_end+1, :2]
        if future_world.shape[0] > 0:
            poly_local = to_local_coords(future_world, pos[i, :2], float(yaw[i]))
            poly_local = np.vstack([np.zeros((1, 2), dtype=np.float32), poly_local])  # prepend origin
        else:
            poly_local = np.zeros((1, 2), dtype=np.float32)

        # corridor widths for this frame
        widths_i = width_curve[i, :max(0, poly_local.shape[0] - 1), 0]  # channel 0

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 8))

        # LiDAR scatter
        if pc.shape[1] >= 3 and pc.size > 0:
            vmin = float(np.nanmin(pc[:, 2])) if np.isfinite(pc[:, 2]).any() else 0.0
            vmax = float(np.nanmax(pc[:, 2])) if np.isfinite(pc[:, 2]).any() else 1.0
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sc = ax.scatter(pc[:, 0], pc[:, 1], c=pc[:, 2],
                            cmap="viridis", norm=norm,
                            s=pc_point_size, edgecolors='none', alpha=0.8, label='LiDAR point cloud')
            # colorbar docked on the right
            cax_pc = fig.add_axes([0.84, 0.15, 0.02, 0.7])
            cbar_pc = fig.colorbar(sc, cax=cax_pc)
            cbar_pc.set_label('LiDAR point cloud height (m)', rotation=270, labelpad=10)
        else:
            ax.scatter(pc[:, 0], pc[:, 1], s=pc_point_size, edgecolors='none', alpha=0.8, label='LiDAR point cloud')

        # Trajectory polyline and corridor boxes
        if poly_local.shape[0] > 1:
            ax.plot(poly_local[:, 0], poly_local[:, 1], 'o-', color='tab:blue', lw=1.5, ms=4, label="Robot's trajectory")

            segs = np.stack([poly_local[:-1], poly_local[1:]], axis=1)  # [J,2,2]
            idx = np.arange(segs.shape[0], dtype=np.float32)
            def forward(x): return x
            def inverse(x): return x
            norm_idx = FuncNorm((forward, inverse), vmin=0., vmax=max(1.0, idx.max()))
            lc = LineCollection(segs, array=idx, cmap=plt.cm.Blues, norm=norm_idx, linewidths=2, linestyles='solid')
            ax.add_collection(lc)

            for j in range(segs.shape[0]):
                w = float(widths_i[j]) if j < len(widths_i) and np.isfinite(widths_i[j]) else 0.0
                if w <= 0.0:
                    continue
                p0 = poly_local[j]
                p1 = poly_local[j + 1]
                poly = segment_rect_polygon(p0, p1, width=w)  # 4x2
                patch = Polygon(poly, closed=True,
                                facecolor='k', edgecolor='k',
                                linewidth=0.6, alpha=corridor_alpha)
                ax.add_patch(patch)

        # Cosmetics
        ax.set_xlabel('X (m)')
        ax.set_xlim([-1.5, 10.0])
        ax.set_ylim([-8.5, 7.5])
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal', 'box')
        ax.set_title('LiDAR point cloud without ground and minimum corridor width per segment', fontsize=15)
        ax.legend(loc='upper left', fontsize=12, frameon=True)

        plt.subplots_adjust(right=0.80)

        out_path = os.path.join(out_dir, f"corridor_{i:06d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Visualize corridor width_curve over LiDAR frames. Default Zarr mode; use --use-tar for legacy *_ng.npy mode."
    )
    ap.add_argument("trajectory_dir", type=str, help="Path to a single trajectory folder")
    ap.add_argument("--out", type=str, default="./corridor_viz", help="Output directory for PNGs")
    ap.add_argument("--max-frames", type=int, default=-1, help="Max number of frames to render (-1 = all)")
    ap.add_argument("--step", type=int, default=1, help="Stride between frames")
    ap.add_argument("--z-max", type=float, default=None, help="Optional |Z| clip for cleaner color scale")
    ap.add_argument("--pc-size", type=float, default=5.0, help="Scatter point size for LiDAR")
    ap.add_argument("--alpha", type=float, default=0.15, help="Alpha for corridor boxes")
    ap.add_argument("--use-tar", action="store_true", help="Legacy mode: read .pkl, width_curve.npy and *_ng.npy files")
    args = ap.parse_args()

    visualize_traj_with_corridors(
        traj_dir=args.trajectory_dir,
        out_dir=args.out,
        max_frames=args.max_frames,
        step=args.step,
        z_clip=args.z_max,
        pc_point_size=args.pc_size,
        corridor_alpha=args.alpha,
        use_tar=args.use_tar,
    )

if __name__ == "__main__":
    main()