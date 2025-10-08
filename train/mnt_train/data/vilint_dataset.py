import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb
import hashlib
import tarfile
import re

from collections import defaultdict

# Point cloud parallel path
from concurrent.futures import ProcessPoolExecutor, as_completed
 # --- Parallel point cloud worker --------------------------------------------
# Reads a batch of times from a Zarr path and returns [(key_bytes, value_bytes), ...]
# Values are npy-encoded (16384,3) float16 arrays ready for LMDB putmulti.
from mnt_train.data.data_utils import pad_point_cloud as _pad_pc_np

#
# Safe batch put that works across python-lmdb versions
def _txn_put_multi(txn, kvs):
    if not kvs:
        return
    pm = getattr(txn, "putmulti", None)
    if pm is not None:
        return pm(kvs, dupdata=False, overwrite=True)
    # Fallback to cursor.putmulti if available; else loop
    try:
        with txn.cursor() as cur:
            return cur.putmulti(kvs, dupdata=False, overwrite=True)
    except Exception:
        for k, v in kvs:
            txn.put(k, v)
        return None

def _pc16k_worker_batch(args):
    import io as _io
    import numpy as _np
    import zarr as _zarr
    from pathlib import Path as _Path

    pc_path, traj_name, times, point_cloud_size, lidar_transform = args
    try:
        arr = _zarr.open_array(_zarr.storage.DirectoryStore(str(_Path(pc_path))), mode="r")
    except Exception:
        return []
    out = []
    for t in times:
        try:
            pc = _np.asarray(arr[int(t)], dtype=_np.float32)
            if pc.ndim != 2 or pc.shape[1] < 2:
                pc16 = _np.zeros((point_cloud_size, 3), dtype=_np.float16)
            else:
                mask = ((pc[:, 0] > 0.0) & (pc[:, 0] < 20.0) &
                        (pc[:, 1] < 20.0) & (pc[:, 1] > -20.0))
                pc = pc[mask]
                if pc.shape[1] == 2:
                    pc = _np.concatenate([pc, _np.zeros((pc.shape[0], 1), dtype=pc.dtype)], axis=1)
                # Use numpy branch of helper to avoid torch overhead in workers
                pc_padded = _pad_pc_np(pc, point_cloud_size)
                pc = np.hstack([pc_padded, np.ones((pc.shape[0], 1))])
                pc = (lidar_transform @ pc.T).T[:, :3]
                pc16 = pc.astype(_np.float16, copy=False)
            # Store raw float16 bytes (C-order) to avoid .npy overhead
            if not pc16.flags["C_CONTIGUOUS"]:
                pc16 = _np.ascontiguousarray(pc16)
            out.append((f"pc16k:{traj_name}:{int(t):06d}".encode(), pc16.tobytes(order="C")))
        except Exception:
            # Skip problematic timestep
            continue
    return out


import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import pandas as pd

from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
import math
from pathlib import Path

from mnt_train.data.data_utils import (
    img_path_to_data,
    lidar_path_to_data,
    pad_point_cloud,
    calculate_sin_cos,
    to_local_coords,
    list_to_local_coords,
    get_collision_status_path,
    get_width_curve_path,
    _open_zarr_array,
    get_image_bytes_from_images_tar,
)

class ViLiNT_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist_cat: int,
        min_action_distance: int,
        max_action_distance: int,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        context_size_li: int, 
        is_lidar: bool,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str = "image",
        goal_type: str = "image",
        distance_type: str = "euclidean",
    ):
        """
        Main ViLiNT dataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each separated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            waypoint_spacing (int): Spacing between waypoints
            min_dist_cat (int): Minimum distance category to use
            max_dist_cat (int): Maximum distance category to use
            negative_mining (bool): Whether to use negative mining from the ViNG paper (Shah et al.) (https://arxiv.org/abs/2012.09812)
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
            obs_type (str): What data type to use for the observation. Only "image" is supported.
            goal_type (str): What data type to use for the goal. The only one supported is "image" for now.
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.negative_mining = negative_mining
        if self.negative_mining:
            self.distance_categories.append(-1)
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle

        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance

        self.context_size_im = context_size
        self.context_size_li = context_size_li
        self.is_lidar = is_lidar
        self.context_size = max(self.context_size_li, self.context_size_im)
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize
        self.obs_type = obs_type
        self.goal_type = goal_type
        self.distance_type = distance_type

        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}
        self._load_index()
        self._build_caches()

        self.robot_config = self._get_robot_config(all_data_config)
        
        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_lidar_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()

    # --- helpers for per-trajectory resources ---------------------------------
    def _traj_has_images_tar(self, traj_name: str) -> bool:
        return os.path.exists(os.path.join(self.data_folder, traj_name, "images.tar"))

    def _points_zarr_path(self, traj_name: str) -> str:
        return os.path.join(self.data_folder, traj_name, "points.zarr")

    def _width_curve_zarr_path(self, traj_name: str) -> str:
        return os.path.join(self.data_folder, traj_name, "width_curve_temporal.zarr")

    def _pose_pos_zarr_path(self, traj_name: str) -> str:
        return os.path.join(self.data_folder, traj_name, "trajectory.zarr/positions.zarr")

    def _pose_orien_zarr_path(self, traj_name: str) -> str:
        return os.path.join(self.data_folder, traj_name, "trajectory.zarr/yaws.zarr")

    def _open_points_zarr_for_traj(self, traj_name: str):
        if not hasattr(self, "_pc_arrays"):
            self._pc_arrays = {}
        if traj_name not in self._pc_arrays:
            self._pc_arrays[traj_name] = _open_zarr_array(Path(self._points_zarr_path(traj_name)))
        return self._pc_arrays[traj_name]

    def _open_width_curve_zarr_for_traj(self, traj_name: str):
        if not hasattr(self, "_wc_arrays"):
            self._wc_arrays = {}
        if traj_name not in self._wc_arrays:
            self._wc_arrays[traj_name] = _open_zarr_array(Path(self._width_curve_zarr_path(traj_name)))
        return self._wc_arrays[traj_name]

    def _orientation_to_yaw(self, ori: np.ndarray) -> np.ndarray:
        ori = np.asarray(ori)
        if ori.ndim == 1:
            return ori.astype(np.float32)
        if ori.ndim == 2 and ori.shape[1] == 1:
            return ori[:, 0].astype(np.float32)
        if ori.ndim == 2 and ori.shape[1] == 3:
            # Assume Euler (roll, pitch, yaw) with yaw last
            return ori[:, 2].astype(np.float32)
        if ori.ndim == 2 and ori.shape[1] == 4:
            # Try both (w,x,y,z) and (x,y,z,w)
            q = ori.astype(np.float64)
            # Heuristic: if last column's magnitude > first, assume (x,y,z,w)
            if np.mean(np.abs(q[:, -1])) > np.mean(np.abs(q[:, 0])):
                x, y, z, w = q.T
            else:
                w, x, y, z = q.T
            # Yaw around Z
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            return yaw.astype(np.float32)
        raise ValueError(f"Unrecognized orientation shape {ori.shape}")
    
    def _load_lidar_to_base_transform(self) -> np.ndarray:
        lidar_to_base_transform = self.data_config["robot"]["lidar_transform"]
        rotation_dict = lidar_to_base_transform["rotation"]
        translation_dict = lidar_to_base_transform["translation"]
        rotation = np.array([rotation_dict["x"], rotation_dict["y"], rotation_dict["z"], rotation_dict["w"]], dtype=np.float64)
        translation = np.array([translation_dict["x"], translation_dict["y"], translation_dict["z"]], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        R_mat = R.from_quat(rotation).as_matrix()
        T[:3, :3] = R_mat
        T[:3, 3] = translation
        return T


    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB.
        Supported image sources: images.tar only. Trajectories from .pkl or pose .zarr. Lidar/width curves from .zarr.
        """
        import io
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}.lmdb",
        )

        # Helpers for array <-> bytes
        def _np_to_bytes(a: np.ndarray) -> bytes:
            buf = io.BytesIO()
            np.save(buf, a, allow_pickle=False)
            return buf.getvalue()

        cached_wc = set()       # trajectories for which width_curve was cached
        cached_traj = set()     # trajectories for which traj/pos/yaw were cached

        # Ensure trajectories are loaded so we know their lengths
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        if not os.path.exists(cache_filename):
            # Deduplicate times per trajectory to avoid reopening/retouching repeatedly
            traj_to_times = defaultdict(set)
            for traj_name, time in self.goals_index:
                traj_to_times[traj_name].add(int(time))

            with lmdb.open(
                cache_filename,
                map_size=2**40,
                writemap=True,
                map_async=True,
                sync=False,
                metasync=False,
                readahead=False,
            ) as cache_env:
                with cache_env.begin(write=True) as txn:
                    outer = tqdm.tqdm(traj_to_times.items(), disable=not use_tqdm, dynamic_ncols=True,
                                      desc=f"Building LMDB cache (all arrays) for {self.dataset_name}")
                    for traj_name, times in outer:
                        # 1) Images: open tar ONCE and index only needed members; then use raw offset reads
                        if self._traj_has_images_tar(traj_name):
                            tar_path = os.path.join(self.data_folder, traj_name, "images.tar")
                            try:
                                need_idxs = set(int(t) for t in times)
                                # Build index of needed members only (saves Python work)
                                with tarfile.open(tar_path, mode="r") as tf:
                                    member_map = {}
                                    for m in tf.getmembers():
                                        if not m.isfile():
                                            continue
                                        base = os.path.basename(m.name)
                                        m0 = re.match(r"^0*([0-9]+)\.(jpe?g|png)$", base, flags=re.IGNORECASE)
                                        if not m0:
                                            continue
                                        i = int(m0.group(1))
                                        if i in need_idxs and i not in member_map:
                                            member_map[i] = m
                                kvs = []
                                # Fast path: raw file reads using TarInfo offsets (works for uncompressed .tar)
                                try:
                                    with open(tar_path, 'rb', buffering=0) as raw:
                                        for t in sorted(times):
                                            info = member_map.get(int(t))
                                            if info is None:
                                                continue
                                            raw.seek(info.offset_data)
                                            data = raw.read(info.size)
                                            kvs.append((f"img:{traj_name}:{int(t):06d}".encode(), data))
                                    if kvs:
                                        _txn_put_multi(txn, kvs)
                                except Exception:
                                    # Fallback: use tarfile API
                                    with tarfile.open(tar_path, mode="r") as tf:
                                        for t in sorted(times):
                                            info = member_map.get(int(t))
                                            if info is None:
                                                continue
                                            fobj = tf.extractfile(info)
                                            if fobj is None:
                                                continue
                                            txn.put(f"img:{traj_name}:{int(t):06d}".encode(), fobj.read())
                            except Exception as e:
                                print(f"[WARN] Images caching failed for {traj_name}: {e}")
                        # 2) Point clouds per time — parallel read & preprocess to 16k float16
                        pc_path = self._points_zarr_path(traj_name)
                        if os.path.exists(pc_path):
                            try:
                                times_sorted = sorted(times)
                                # Batch times to amortize worker startup; default 64, override via env
                                batch_size = int(os.environ.get("MNT_PC_BATCH", "64"))
                                worker_count = int(os.environ.get("MNT_PC_WORKERS", max(1, (os.cpu_count() or 8)//2)))
                                batches = [times_sorted[i:i+batch_size] for i in range(0, len(times_sorted), batch_size)]
                                # Process in parallel but keep a single LMDB writer
                                with ProcessPoolExecutor(max_workers=worker_count) as ex:
                                    futures = [ex.submit(_pc16k_worker_batch, (pc_path, traj_name, b, 16384, self._load_lidar_to_base_transform())) for b in batches]
                                    for fut in as_completed(futures):
                                        kvs_pc = fut.result()
                                        if kvs_pc:
                                            _txn_put_multi(txn, kvs_pc)
                            except Exception as e:
                                print(f"[WARN] PC caching error for {traj_name}: {e}")
                        # 3) width_curve once per traj
                        if traj_name not in cached_wc:
                            wc_path = self._width_curve_zarr_path(traj_name)
                            if os.path.exists(wc_path):
                                try:
                                    wc_arr = _open_zarr_array(Path(wc_path))
                                    wc_np = np.asarray(wc_arr, dtype=np.float32)
                                    wc_np = np.ascontiguousarray(wc_np)
                                    txn.put(f"wc:{traj_name}".encode(), wc_np.tobytes(order="C"))
                                    txn.put(
                                        f"wc_shape:{traj_name}".encode(),
                                        np.asarray(wc_np.shape, dtype=np.int32).tobytes(),
                                    )
                                except Exception as e_wc:
                                    print(f"[WARN] Skipping width_curve {traj_name}: {e_wc}")
                            cached_wc.add(traj_name)
                        # 4) Trajectory arrays once per traj
                        if traj_name not in cached_traj:
                            traj_dir = os.path.join(self.data_folder, traj_name)
                            pkl_path = os.path.join(traj_dir, "traj_data.pkl")
                            try:
                                if os.path.exists(pkl_path):
                                    with open(pkl_path, 'rb') as f:
                                        txn.put(f"traj:{traj_name}".encode(), f.read())
                                else:
                                    pos_arr = _open_zarr_array(Path(self._pose_pos_zarr_path(traj_name)))
                                    ori_arr = _open_zarr_array(Path(self._pose_orien_zarr_path(traj_name)))

                                    pos = np.asarray(pos_arr, dtype=np.float32)[:, :2]
                                    yaw = np.asarray(ori_arr, dtype=np.float32)

                                    pos = np.ascontiguousarray(pos)
                                    yaw = np.ascontiguousarray(yaw)

                                    txn.put(f"pos:{traj_name}".encode(), pos.tobytes(order="C"))
                                    txn.put(f"pos_shape:{traj_name}".encode(), np.asarray(pos.shape, dtype=np.int32).tobytes())

                                    txn.put(f"yaw:{traj_name}".encode(), yaw.tobytes(order="C"))
                                    txn.put(f"yaw_shape:{traj_name}".encode(), np.asarray(yaw.shape, dtype=np.int32).tobytes())
                            except Exception as e_tr:
                                print(f"[WARN] Skipping trajectory cache for {traj_name}: {e_tr}")
                            cached_traj.add(traj_name)

        # Reopen the cache in read-only mode
        self._image_lidar_cache: lmdb.Environment = lmdb.open(cache_filename, max_readers=2048, readonly=True, lock=False)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, (traj_len - curr_time - 1)//self.waypoint_spacing)
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time):
        """
        Load an image for the given trajectory and timestep.
        Supported sources: LMDB cache or images.tar only.
        """
        lmdb_key = f"img:{trajectory_name}:{int(time):06d}".encode()
        try:
            with self._image_lidar_cache.begin() as txn:
                image_buffer = txn.get(lmdb_key)
            if image_buffer is not None:
                return img_path_to_data(io.BytesIO(bytes(image_buffer)), self.image_size)
        except Exception:
            pass
        # Fallback: direct tar only
        if self._traj_has_images_tar(trajectory_name):
            img_bytes = get_image_bytes_from_images_tar(os.path.join(self.data_folder, trajectory_name), int(time))
            return img_path_to_data(io.BytesIO(img_bytes), self.image_size)
        raise FileNotFoundError(f"images.tar not found for trajectory {trajectory_name}; only TAR-based images are supported.")
    
    def _load_point_cloud(self, trajectory_name, time):
        try:
            # 1) Try compact preprocessed version
            with self._image_lidar_cache.begin() as txn:
                buf16 = txn.get(f"pc16k:{trajectory_name}:{int(time):06d}".encode())
            if buf16 is not None:
                # LMDB returns a read-only buffer; torch.frombuffer warns on non-writable buffers.
                mv = memoryview(bytearray(buf16))
                t16 = torch.frombuffer(mv, dtype=torch.float16).reshape(16384, 3)
                return t16.to(dtype=torch.float32).contiguous()

            # 2) Try raw version from LMDB, then preprocess
            with self._image_lidar_cache.begin() as txn:
                buf = txn.get(f"pc:{trajectory_name}:{int(time):06d}".encode())
            if buf is not None:
                pc = np.load(io.BytesIO(bytes(buf)), allow_pickle=False)
            else:
                # 3) Fallback: read from Zarr
                pc_arr = self._open_points_zarr_for_traj(trajectory_name)
                pc = pc_arr[int(time)].astype(np.float32, copy=False)
                # ROI crop and padding to 16384 points
                if pc.ndim != 2 or pc.shape[1] < 2:
                    return torch.zeros((16384, 3), dtype=torch.float32)
                mask = (
                    (pc[:, 0] > 0.0) & (pc[:, 0] < 20.0) &
                    (pc[:, 1] < 20.0) & (pc[:, 1] > -20.0)
                )
                pc = pc[mask]
                if pc.shape[1] == 2:
                    pc = np.concatenate([pc, np.zeros((pc.shape[0], 1), dtype=pc.dtype)], axis=1)
            pc_t = torch.from_numpy(pc.copy()).float()
            return pad_point_cloud(pc_t, 16384)
        except Exception as e:
            print(f"Failed to load point cloud for {trajectory_name}[{time}]: {e}")
            return torch.zeros((16384, 3), dtype=torch.float32)

    def _load_collision_status(self, trajectory_name):
        collision_status_path = get_collision_status_path(self.data_folder, trajectory_name)
        try:
            with self._image_lidar_cache.begin() as txn:
                coll_buffer = txn.get(collision_status_path.encode())
                coll_bytes = bytes(coll_buffer)
            coll_bytes = io.BytesIO(coll_bytes)
            return torch.from_numpy(np.load(coll_bytes, allow_pickle=False))
        except TypeError:
            print(f"Failed to load collision status {collision_status_path}")
    
    def _load_width_curve(self, trajectory_name):
        try:
            with self._image_lidar_cache.begin() as txn:
                buf = txn.get(f"wc:{trajectory_name}".encode())
                buf_shape = txn.get(f"wc_shape:{trajectory_name}".encode())

            if buf is not None:
                b = bytes(buf)
                # New fast path: raw float32 bytes + stored shape
                if buf_shape is not None:
                    shape = tuple(np.frombuffer(bytes(buf_shape), dtype=np.int32).tolist())
                    # LMDB returns a read-only buffer; make it writable for torch.frombuffer
                    mv = memoryview(bytearray(b))
                    wc_t = torch.frombuffer(mv, dtype=torch.float32).reshape(*shape)
                    return wc_t

                # If shape missing, fall back to Zarr
            wc_arr = self._open_width_curve_zarr_for_traj(trajectory_name)
            wc_np = np.asarray(wc_arr, dtype=np.float32)
            return torch.from_numpy(wc_np)
        except Exception as e:
            print(f"Failed to load width curves for {trajectory_name}: {e}")
            return torch.full((self.len_traj_pred, 2), float('inf'))

    def _stable_k(self, traj_name: str, curr_time: int) -> int:
        """Deterministic {1,2,3} from a hash of (traj_name, curr_time).
        """
        h = hashlib.blake2b(f"{traj_name}:{curr_time}".encode(), digest_size=1).digest()[0]
        return 1 + (h % 2)

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
        goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        if self.learn_angle:
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            actions = waypoints[1:]
        
        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos
    
    def _get_collision_status(self, collisions, curr_time, apply_cummin=None):
        """
        Build a per-waypoint [len_traj_pred, 2] tensor of corridor metrics where
        column 0 is width and column 1 is length.

        By default a cumulative minimum is applied along the time/lookahead axis
        (same behavior as before). You can disable this by either:
        - passing `apply_cummin=False` at call time, or
        - setting an instance attribute `self.collision_use_cummin = False`.

        Args:
            collisions (torch.Tensor): shape [T, H, 2] from width_curve.npy
            curr_time (int): current time index for the observation
            apply_cummin (bool or None):
                * None  -> use getattr(self, "collision_use_cummin", True)
                * True  -> apply cumulative minimum along time
                * False -> keep raw per-lookahead values
        Returns:
            torch.Tensor: shape [len_traj_pred, 2] corridor metrics
        """
        assert isinstance(collisions, torch.Tensor), "collisions must be a torch.Tensor"
        assert collisions.ndim == 3 and collisions.size(-1) == 2, \
            f"Expected [T, H, 2], got {tuple(collisions.shape)}"

        # Resolve behavior flag
        if apply_cummin is None:
            apply_cummin = getattr(self, "collision_use_cummin", True)

        # We index from the next step (lookahead of 1)
        start_index = curr_time + 1
        span = self.len_traj_pred * self.waypoint_spacing

        T, H, _ = collisions.shape
        if start_index >= T:
            # Not enough future info; return +inf to indicate no constraint
            return torch.full((self.len_traj_pred, 2), float('inf'))

        # 1) Start with the whole horizon from the current step
        seq_list = [collisions[start_index]]  # [H, 2]

        # 2) Extend beyond the horizon by appending the last lookahead (-1)
        max_tail_needed = span - 1  # we already have the first element from start_index
        max_tail_avail = max(0, T - (start_index + 1))
        tail_len = min(max_tail_needed, max_tail_avail)
        if tail_len > 0:
            tails = [collisions[start_index + t, -1, :].unsqueeze(0) for t in range(1, tail_len + 1)]
            seq_list.append(torch.cat(tails, dim=0))

        seq = torch.cat(seq_list, dim=0)  # shape [H + tail_len, 2]

        # 3) Ensure we have at least `span` entries; if not, pad with the last value
        if seq.size(0) < span:
            pad = seq[-1:].repeat(span - seq.size(0), 1)
            seq = torch.cat([seq, pad], dim=0)

        # Keep exactly the next `span` entries
        seq = seq[:span]

        # 4) Optionally apply cumulative minimum along time dimension
        if apply_cummin:
            cum_w = torch.cummin(seq[:, 0], dim=0)[0]
            cum_l = torch.cummin(seq[:, 1], dim=0)[0]
            seq_out = torch.stack([cum_w, cum_l], dim=1)  # [span, 2]
        else:
            seq_out = seq  # keep raw values

        # 5) Subsample according to waypoint_spacing to get exactly len_traj_pred rows
        idxs = torch.arange(0, span, self.waypoint_spacing)
        seq_out = seq_out.index_select(0, idxs)
        if seq_out.size(0) < self.len_traj_pred:
            last = seq_out[-1:].repeat(self.len_traj_pred - seq_out.size(0), 1)
            seq_out = torch.cat([seq_out, last], dim=0)
        else:
            seq_out = seq_out[:self.len_traj_pred]

        return seq_out
    
    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        # Try LMDB shortcuts first
        try:
            with self._image_lidar_cache.begin() as txn:
                b_traj = txn.get(f"traj:{trajectory_name}".encode())
            if b_traj is not None:
                traj_data = pickle.load(io.BytesIO(bytes(b_traj)))
                self.trajectory_cache[trajectory_name] = traj_data
                return traj_data
            with self._image_lidar_cache.begin() as txn:
                b_pos = txn.get(f"pos:{trajectory_name}".encode())
                b_pos_shape = txn.get(f"pos_shape:{trajectory_name}".encode())
                b_yaw = txn.get(f"yaw:{trajectory_name}".encode())
                b_yaw_shape = txn.get(f"yaw_shape:{trajectory_name}".encode())   
            if (b_pos is not None) and (b_yaw is not None):
                pos_shape = tuple(np.frombuffer(b_pos_shape, dtype=np.int32).tolist())
                yaw_shape = tuple(np.frombuffer(b_yaw_shape, dtype=np.int32).tolist())
                pos = np.frombuffer(b_pos, dtype=np.float32).reshape(*pos_shape)
                yaw = np.frombuffer(b_yaw, dtype=np.float32).reshape(*yaw_shape)
                traj_data = {"position": pos.astype(np.float32, copy=False), "yaw": yaw.astype(np.float32, copy=False)}
                self.trajectory_cache[trajectory_name] = traj_data
                return traj_data
        except Exception:
            pass
        traj_dir = os.path.join(self.data_folder, trajectory_name)
        pkl_path = os.path.join(traj_dir, "traj_data.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data
        # Otherwise, try pose_pos.zarr and pose_orien.zarr
        pos_path = self._pose_pos_zarr_path(trajectory_name)
        ori_path = self._pose_orien_zarr_path(trajectory_name)
        assert os.path.exists(pos_path), f"Missing pose_pos.zarr in {traj_dir}"
        assert os.path.exists(ori_path), f"Missing pose_orien.zarr in {traj_dir}"
        pos_arr = _open_zarr_array(Path(pos_path))
        ori_arr = _open_zarr_array(Path(ori_path))
        pos = np.asarray(pos_arr)
        # Keep XY; if already 2D use as-is
        if pos.ndim != 2 or pos.shape[1] < 2:
            raise ValueError(f"pose_pos.zarr has unexpected shape {pos.shape}")
        position_xy = pos[:, :2].astype(np.float32)
        yaw = self._orientation_to_yaw(np.asarray(ori_arr))
        traj_data = {"position": position_xy, "yaw": yaw}
        self.trajectory_cache[trajectory_name] = traj_data
        return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)
    

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to the i-th datapoint
        Returns:
            Tuple of tensors containing:
                - obs_image (torch.Tensor): tensor of shape [3, H, W] with the observation image
                - goal_image (torch.Tensor): tensor of shape [3, H, W] with the goal image 
                - actions_torch (torch.Tensor): tensor of shape (N, 2) or (N, 3) (if training with angle) with the action labels
                - distance (torch.Tensor): tensor containing the distance label
                - goal_pos (torch.Tensor): tensor with the goal position
                - dataset_index (torch.Tensor): index of the dataset (for visualization/multiple datasets)
                - action_mask (torch.Tensor): binary mask indicating valid actions
                - obs_point_cloud (torch.Tensor): tensor containing the concatenated 3D point clouds associated with the observation images
                - goal_point_cloud (torch.Tensor): tensor containing the 3D point cloud associated with the goal image
        """
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)

        # Load images
        if self.context_type == "temporal":
            # sample the last self.context_size times from interval [curr_time - self.context_size * spacing, curr_time]
            context_times_im = list(
                range(
                    curr_time - self.context_size_im * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context_im = [(f_curr, t) for t in context_times_im]
            context_times_li = list(
                range(
                    curr_time - self.context_size_li * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context_li = [(f_curr, t) for t in context_times_li]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context_im
        ])

        # Load corresponding 3D point clouds
        if self.is_lidar:
            obs_point_cloud = torch.stack([self._load_point_cloud(f, t) for f, t in context_li], dim=0)
        else:
            obs_point_cloud = torch.zeros((len(context_li), 16384, 3), dtype=torch.float32)

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} and {goal_traj_len}"

        # Compute actions
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
        
        # Compute distances
        if goal_is_negative:
            distance = self.max_dist_cat
        elif self.distance_type == "temporal":
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
        elif self.distance_type == "euclidean":
            positions_segment = curr_traj_data["position"][curr_time:goal_time + 1]
            segment_diffs = np.diff(positions_segment, axis=0)
            segment_lengths = np.linalg.norm(segment_diffs, axis=1)
            distance = np.sum(segment_lengths)
        
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)

    
        actions_xy = torch.as_tensor(actions[:, :2], dtype=torch.float32)
        waypoints_xy = torch.cumsum(actions_xy, dim=0)
        # ROI: x in [0, 10], y in [-10, 10]
        roi_mask = (
            (waypoints_xy[:, 0] >= 0.0) & (waypoints_xy[:, 0] <= 10.0) &
            (waypoints_xy[:, 1] >= -10.0) & (waypoints_xy[:, 1] <= 10.0)
        ).to(torch.float32)

        action_mask = (
            (distance < self.max_action_distance) and
            (distance > self.min_action_distance) and
            (not goal_is_negative)
        )

        # Load collision status
        # robot params: [lidar_height, width, height, length, ground_clearance]
        # Canonical robot for labels (keeps targets stationary)
        robot_canon = torch.cat((self.robot_config[:2], self.robot_config[-1].unsqueeze(0)))

        # Deterministic augmentation factor k \in {1,2,3}
        # k = self._stable_k(f_curr, curr_time)
        k1 = 2.0 * torch.rand(()) + 0.8
        k2 = 2.0 * torch.rand(()) + 0.8

        # Physics fed to the net (augmented); labels remain canonical
        physics_for_net = robot_canon.clone()
        physics_for_net[0] = k1 * self.robot_config[0]   # width
        physics_for_net[1] = k2 * self.robot_config[1]   # length

        width_curve = self._load_width_curve(f_curr)
        width_curve_traj = self._get_collision_status(width_curve, curr_time, apply_cummin=False)
        width_curve_traj = torch.nan_to_num(width_curve_traj, nan=float(20.0), posinf=float(20.0), neginf=float(20.0))
        col_ratio = torch.min(width_curve_traj[..., 0] / (physics_for_net[0] + 1e-6), width_curve_traj[..., 0] / (physics_for_net[1] + 1e-6))
        col_mask = roi_mask # per-waypoint mask used to zero-out BCE terms outside LiDAR ROI
        # action_mask = action_mask and (col_status.sum() <= self.len_traj_pred * 0.5)

        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            obs_point_cloud,
            torch.as_tensor(goal_pos, dtype=torch.float32),    
            physics_for_net,
            actions_torch,
            torch.as_tensor(distance, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
            torch.as_tensor(col_ratio, dtype=torch.float32),
            col_mask,
            self.is_lidar,
        )
    
    def _get_robot_config(self, data_config):
        robot_config = data_config[self.dataset_name].get('robot')
        assert robot_config is not None, f"No robot parameters in the chosen dataset: {self.dataset_name}"
        lidar_height = robot_config['lidar_height']
        width = robot_config['width']
        height = robot_config['height']
        length = robot_config['length']
        ground_clearance = robot_config['ground_clearance']
        max_speed = robot_config['max_speed']
        max_yaw_rate = robot_config['max_yaw_rate']
        max_acceleration = robot_config['max_acceleration']

        # robot_params_tensor = torch.tensor([lidar_height, width, height, length, ground_clearance], dtype=torch.float32)
        robot_params_tensor = torch.tensor([width, length, max_speed, max_yaw_rate, max_acceleration, lidar_height], dtype=torch.float32)

        return robot_params_tensor

