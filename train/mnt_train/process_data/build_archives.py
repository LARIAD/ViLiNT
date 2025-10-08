#!/usr/bin/env python3
"""
Build per-trajectory image .tar and LiDAR .zarr aligned by numeric index.

Expected dataset layout:
    dataset/
      datas/
        trajectories/
          <traj_1>/
            0.jpg, 1.jpg, ..., N.jpg
            0.npy, 1.npy, ..., M.npy
          <traj_2>/
            ...
          ...

Outputs (inside each trajectory folder):
    images.tar                # all JPEGs at tar root as <idx>.jpg
    points.zarr               # Zarr array stacked as (N,P,3), chunks=(256, min(P,200000), 3)
# (note) while scanning *.npy, exclude *_ng.npy, width_curve*.npy, clearance*.npy, ttc*.npy, collision*.npy
    aligned_indices.txt       # indices used for Zarr (one per line)
    build_summary.json        # small summary for this trajectory

Usage:
    python build_archives.py --root dataset/datas/trajectories
Options:
    --align {intersection,points,images} :
        intersection (default): Zarr frames = indices present in BOTH images & npy
        points:                 Zarr frames = indices present in npy only
        images:                 Zarr frames = indices present in images only
    --only-aligned-images     : Make images.tar include only aligned indices (off by default)
    --overwrite               : Overwrite existing outputs
    --image-exts              : Comma-separated list, default: jpg,jpeg
"""

import argparse
import json
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import zarr
import zarr.codecs as zc
from datetime import datetime


def _make_out_array(out_path: Path, shape, dtype, chunks=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if chunks is None:
        c0 = min(64, shape[0]) if shape else None
        if c0 is not None and len(shape) >= 1:
            chunks = (max(1, c0),) + tuple(min(int(s), 256) for s in shape[1:])
        else:
            chunks = True  # if you rely on "auto", consider making this a real tuple for v3

    codecs = [
        zc.BytesCodec(),  # array->bytes serializer (default is usually fine)
        zc.BloscCodec(cname="zstd", clevel=5, shuffle=zc.BloscShuffle.shuffle),
    ]

    return zarr.create(
        store=str(out_path),
        shape=shape,
        chunk_shape=chunks if chunks is not True else None,  # v3 wants an explicit shape
        dtype=dtype,
        codecs=codecs,
        overwrite=True,
        zarr_format=3,
    )


def numeric_index(path: Path) -> int:
    """
    Extract numeric index from filename stem, e.g. '000123.jpg' -> 123, '42.npy' -> 42.
    Raises ValueError if no integer is found.
    """
    s = path.stem
    # Prefer the entire stem if it is an int; else grab the last integer substring.
    if re.fullmatch(r"\d+", s):
        return int(s)
    m = re.search(r"(\d+)$", s)
    if not m:
        raise ValueError(f"No trailing integer in filename: {path.name}")
    return int(m.group(1))


def collect_by_index(dirpath: Path, extensions: List[str]) -> Dict[int, Path]:
    """Return {index -> file_path} for files in dirpath with given extensions (case-insensitive), excluding preprocessed/supervision npy files."""
    result = {}
    exts = {e.lower() for e in extensions}
    for p in dirpath.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") in exts:
            name = p.name.lower()
            # Skip preprocessed/supervision npy files explicitly
            if name.endswith("_ng.npy") or "width_curve" in name or "clearance" in name or name.startswith("ttc") or "ttc_" in name or "collision" in name:
                continue
            try:
                idx = numeric_index(p)
                result[idx] = p
            except ValueError:
                # silently skip non-indexed files
                pass
    return result
def _points_count_fast(npy_path: Path) -> int:
    """Return number of points for a .npy storing (N,3) or (3,N) without loading the full array."""
    arr = np.load(npy_path, allow_pickle=False, mmap_mode='r')
    if arr.ndim != 2:
        return 0
    if arr.shape[1] >= 3:
        return int(arr.shape[0])
    if arr.shape[0] >= 3 and arr.shape[1] < 3:
        return int(arr.shape[1])
    return 0



def load_points_fix_shape(npy_path: Path) -> np.ndarray:
    """
    Load a .npy point cloud as float32 (N,3) XYZ.
    Accepts (N,3) or (3,N); if more than 3 cols, keeps first 3.
    """
    arr = np.load(npy_path, allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"{npy_path} is not 2D (got shape {arr.shape})")

    # Decide orientation:
    if arr.shape[1] >= 3:
        pts = arr[:, :3]
    elif arr.shape[0] >= 3 and arr.shape[1] < 3:
        pts = arr[:3, :].T  # (3,N) -> (N,3)
    else:
        raise ValueError(f"{npy_path} cannot be interpreted as XYZ (shape {arr.shape})")

    return pts.astype(np.float32, copy=False)


def build_images_tar(image_map: Dict[int, Path],
                     out_tar: Path,
                     topic: str = "images",
                     indices: List[int] = None,
                     overwrite: bool = False):
    
    use_indices = sorted(indices) if indices is not None else sorted(image_map.keys())
    if not use_indices:
        print(f"[WARN] No images to pack into {out_tar}")
        return
    
    if out_tar.exists():
        if overwrite:
            out_tar.unlink()
        else:
            print(f"[SKIP] {out_tar} exists. Use --overwrite to rebuild.")
            return

    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, mode="w") as tf:
        for idx in use_indices:
            p = image_map.get(idx, None)
            if p is None:
                continue  # if filtering by indices, skip missing
            arcname = f"{idx:06d}{p.suffix.lower()}"
            tf.add(p, arcname=arcname)
    print(f"[OK] Wrote {out_tar} with {len(use_indices)} files at tar root.")



def _rm_zarr_path(p: Path):
    """Remove an existing Zarr array directory path recursively if it exists."""
    if p.exists():
        if p.is_file():
            p.unlink()
            return
        for root, dirs, files in os.walk(p, topdown=False):
            for name in files:
                try:
                    os.remove(Path(root) / name)
                except FileNotFoundError:
                    pass
            for name in dirs:
                try:
                    os.rmdir(Path(root) / name)
                except OSError:
                    pass
        try:
            os.rmdir(p)
        except OSError:
            pass


# --- width_curve.npy -> width_curve.zarr helper ---
def convert_width_curve_npy_to_zarr(traj_dir: Path, overwrite: bool) -> None:
    """
    If a width_curve.npy file exists in the trajectory folder, convert it to a
    width_curve.zarr array in the same folder.

    The .npy is loaded as float32 and written as a Zarr array with a single
    chunk equal to its full shape.
    """
    npy_path = traj_dir / "width_curve.npy"
    if not npy_path.exists():
        return

    zarr_path = traj_dir / "width_curve.zarr"
    if zarr_path.exists():
        if not overwrite:
            print(f"[SKIP] {zarr_path} exists. Use --overwrite to rebuild width_curve.zarr.")
            return
        _rm_zarr_path(zarr_path)

    try:
        arr = np.load(npy_path, allow_pickle=False)
    except Exception as e:
        print(f"[WARN] Failed to load {npy_path}: {e}")
        return

    # Ensure float32 for consistency
    arr = np.asarray(arr, dtype=np.float32)
    shape = arr.shape
    if shape == ():
        print(f"[WARN] {npy_path} is scalar; not writing width_curve.zarr.")
        return

    chunks = shape
    d_arr = _make_out_array(zarr_path, shape=shape, dtype=arr.dtype, chunks=chunks)
    d_arr[...] = arr

    try:
        d_arr.attrs.update({
            "source": npy_path.name,
            "note": "width_curve converted from .npy by build_archives.py",
        })
    except Exception:
        pass

    print(f"[OK] Wrote width_curve.zarr at {zarr_path} with shape {shape}.")


def build_points_zarr(points_map: Dict[int, Path],
                      out_points: Path,
                      selected_indices: List[int],
                      overwrite: bool = False):
    """
    Build stacked point clouds into a single 3-D Zarr array:
      - points.zarr: (N,P,3) float32, chunks=(min(256,N), min(P,200000), 3)
    If per-frame point counts vary, we pad with zeros up to P = max frame size.
    """
    if not selected_indices:
        print(f"[WARN] No frames selected for points Zarr under {out_points.parent}")
        return

    # Overwrite handling
    if out_points.exists():
        if overwrite:
            _rm_zarr_path(out_points)
        else:
            print(f"[SKIP] {out_points} exists. Use --overwrite to rebuild.")
            return

    # Inspect sizes and determine P (max points across selected frames) using fast count
    sizes = []
    for idx in selected_indices:
        sizes.append(_points_count_fast(points_map[idx]))
    N = len(selected_indices)
    print(N)
    P = int(max(sizes)) if sizes else 0
    print(P)
    if P <= 0:
        print(f"[WARN] All frames empty for {out_points}; nothing written.")
        return

    C = 3
    chunks = (min(256, N), min(P, 200000), C)
    d_points = _make_out_array(out_points, shape=(N, P, C), dtype=np.float32, chunks=chunks)
    try:
        d_points.attrs.update({
            "layout": "stacked (N,P,3)",
            "chunks": tuple(int(x) for x in chunks),
            "note": "rows beyond per-frame length are zero-padded",
        })
    except Exception:
        pass

    # Fill data (pad/truncate per frame if needed)
    for i, idx in enumerate(selected_indices):
        pts = load_points_fix_shape(points_map[idx])  # (Ni,3)
        n = min(int(pts.shape[0]), P)
        if n > 0:
            d_points[i, :n, :] = pts[:n]
            # print("i: ", i)
        # if n < P:
        #     # zero padding already default; explicit clear for safety
        #     d_points[i, n:P, :] = 0
        #     print("i: ", i)

    print(f"[OK] Wrote stacked points to {out_points} with shape {(N, P, C)} and chunks {chunks}.")


def process_trajectory(traj_dir: Path,
                       align_mode: str,
                       only_aligned_images: bool,
                       overwrite: bool,
                       image_exts: List[str]) -> dict:
    images = collect_by_index(traj_dir, image_exts)
    points = collect_by_index(traj_dir, ["npy"])

    img_indices = set(images.keys())
    pts_indices = set(points.keys())

    if align_mode == "intersection":
        aligned = sorted(img_indices & pts_indices)
    elif align_mode == "points":
        aligned = sorted(pts_indices)
    elif align_mode == "images":
        aligned = sorted(img_indices)
    else:
        raise ValueError("--align must be one of {intersection,points,images}")

    # Outputs
    out_tar = traj_dir / "images.tar"
    out_points = traj_dir / "points.zarr"

    # Build images tar (all images by default, or only aligned if requested)
    tar_indices = aligned if only_aligned_images else None
    build_images_tar(images, out_tar, topic="images", indices=tar_indices, overwrite=overwrite)

    # Build zarr from aligned indices that actually exist in points_map
    aligned_for_points = [i for i in aligned if i in points]
    build_points_zarr(points,
                      out_points=out_points,
                      selected_indices=aligned_for_points,
                      overwrite=overwrite)

    # Convert width_curve.npy -> width_curve.zarr if present
    convert_width_curve_npy_to_zarr(traj_dir, overwrite=overwrite)

    # Write sidecar files
    (traj_dir / "aligned_indices.txt").write_text("\n".join(str(i) for i in aligned_for_points) + "\n")

    summary = {
        "trajectory": str(traj_dir),
        "num_images": len(images),
        "num_points": len(points),
        "align_mode": align_mode,
        "aligned_frames": len(aligned_for_points),
        "images_tar": str(out_tar),
        "points_zarr": str(out_points),
        "only_aligned_images": only_aligned_images,
        "missing_in_images": sorted(map(int, pts_indices - img_indices)),
        "missing_in_points": sorted(map(int, img_indices - pts_indices)),
    }
    (traj_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Pack per-trajectory images (.tar) and points (.zarr).")
    ap.add_argument("--root", type=str, required=True,
                    help="Path to dataset/datas/trajectories directory.")
    ap.add_argument("--align", type=str, default="intersection",
                    choices=["intersection", "points", "images"],
                    help="How to pick frames for Zarr alignment.")
    ap.add_argument("--only-aligned-images", action="store_true",
                    help="If set, images.tar will include only aligned frames (default: include all JPEGs).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing outputs if present.")
    ap.add_argument("--image-exts", type=str, default="jpg,jpeg",
                    help="Comma-separated image extensions to include (default: jpg,jpeg).")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERR] Root {root} does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    trajectories = [p for p in root.iterdir() if p.is_dir()]
    if not trajectories:
        print(f"[WARN] No trajectory folders found under {root}")
        sys.exit(0)

    image_exts = [e.strip().lstrip(".").lower() for e in args.image_exts.split(",") if e.strip()]
    totals = {"processed": 0, "frames": 0, "images": 0, "points": 0}

    print(f"[INFO] Found {len(trajectories)} trajectories under {root}")
    summaries = []
    for traj in sorted(trajectories):
        print(f"\n[INFO] Processing trajectory: {traj.name}")
        summary = process_trajectory(
            traj_dir=traj,
            align_mode=args.align,
            only_aligned_images=args.only_aligned_images,
            overwrite=args.overwrite,
            image_exts=image_exts,
        )
        summaries.append(summary)
        totals["processed"] += 1
        totals["frames"] += summary["aligned_frames"]
        totals["images"] += summary["num_images"]
        totals["points"] += summary["num_points"]

    # Dataset-level summary
    dataset_summary = {
        "root": str(root),
        "trajectories": len(trajectories),
        "total_aligned_frames": totals["frames"],
        "total_images_found": totals["images"],
        "total_points_found": totals["points"],
        "align_mode": args.align,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    (root / "dataset_build_summary.json").write_text(json.dumps(dataset_summary, indent=2))

    print("\n[DONE]")
    print(json.dumps(dataset_summary, indent=2))


if __name__ == "__main__":
    main()