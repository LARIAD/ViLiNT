# Multimodal Navigation Transformer

This repository contains a navigation model that predicts short waypoint trajectories from multimodal observations. The main model used here is ViLiNT, a diffusion-based policy conditioned on image context, 3D observations, robot state, and a goal direction.

For environment and installation instructions, see [SETUP.md](SETUP.md).

## 1. ViLiNT

ViLiNT takes as input:

- a short history of RGB images,
- a 3D observation stream,
- a robot's embodiment vector,
- a goal direction in the robot frame.

It outputs a short trajectory of future position waypoints that can then be converted into velocity commands by a PD controller.

### Encoders used

- Image encoder: `DUNE` ViT.
- 3D encoder: point cloud encoder `PointTransformerV3`.

### How the 3D modality is introduced

The 3D input is first encoded by the point-cloud backbone. Then it is converted into a small set of LiDAR tokens with a tokenizer that groups the scene into angular sectors and radial rings. These 3D tokens are concatenated with the image-history tokens, physics token, and goal token before multimodal fusion with the transformer.

## 2. How to train the model

The main training entry point is:

```bash
cd train
python train.py --config config/vilint.yaml
```

With Slurm:

```bash
cd train
sbatch train.sh
```

### Main training config

The main file to edit is:

- `train/config/vilint.yaml`

### Expected dataset format

The training loader in `train/mnt_train/data/vilint_dataset.py` does not train directly from loose `jpg` / `npy` files. The final per-trajectory format is archive-based:

- `images.tar` for RGB frames
- `points.zarr` for stacked point clouds
- `traj_data.pkl` or pose arrays under `trajectory.zarr/...` for robot poses
- `width_curve.zarr` for the clearance distance ground truth.

The train/test split folders referenced in `train/config/vilint.yaml` should point to splits whose trajectory names match these archived trajectory folders.

## 3. How to deploy with ROS

The ROS2 deployment entry point is:

```bash
cd deployment/src
python3 deploy_vilint.py --model vilint --imgwaypoints
```

The full tmux launcher is:

```bash
cd deployment/src
bash deploy_vilint.sh
```

This starts:

- the ViLiNT inference node,
- the PD waypoint controller,
- RViz,
- a rosbag recorder.

### Files to edit before deployment

- `deployment/config/models.yaml`
  - set `ckpt_path` to the chosen checkpoint to deploy,
  - enable or disable `mask_image`, `mask_lidar`, and heuristics.

- `deployment/config/robot.yaml`
  - set robot velocity limits,
  - set control topic names,
  - adjust robot dimensions if needed.

- `deployment/src/topics_names.py`
  - set the subscribed ROS topics:
    - `IMAGE_TOPIC`
    - `LIDAR_TOPIC`
    - `ODOM_TOPIC`
    - `GOAL_TOPIC`

### Deployment behavior

`deploy_vilint.py` reads the trained model checkpoint, subscribes to image / LiDAR / odometry / goal topics, predicts waypoints, and publishes them on the waypoint topic. `pd_controller.py` then converts the waypoint stream into velocity commands.

### Docker image

A docker image can be found in ```test/docker```. It allows the user to deploy the model alongside the IsaacSim simulation running with ROS2. It is made for x86 architectures equiped with NVIDIA gpus.

NB: The image build is computationnaly intensive because of the flash-attention building process.

## 4. How to generate data from rosbags

There are two extraction scripts:

- ROS1 bags: `train/mnt_train/process_data/process_bags.py`
- ROS2 bags: `train/mnt_train/process_data/process_bags_ros2.py`

These scripts first extract each trajectory into a folder of raw files such as:

- `traj_data.pkl`
- `0.jpg`, `1.jpg`, ...
- `0.npy`, `1.npy`, ...

Then generate the width curves ground truth with:
```bash
cd train/mnt_train/process_data
python process_lidar_collision.py /path/to/dataset/datas/trajectories -d dataset_name
```

After that, you should build the archive format used by `ViLiNT_Dataset` with:

```bash
cd train/mnt_train/process_data
python build_archives.py --root /path/to/dataset/datas/trajectories --overwrite
```

`build_archives.py` creates, inside each trajectory folder:

- `images.tar`
- `points.zarr`
- `aligned_indices.txt`
- `build_summary.json`

and converts `width_curve.npy` to `width_curve.zarr` when present.

### ROS2 example

```bash
cd train/mnt_train/process_data
python process_bags_ros2.py \
  --dataset-name husky \
  --input-dir /path/to/ros2_bags \
  --output-dir /path/to/processed_dataset \
  --sample-rate 4.0
```

### ROS1 example

```bash
cd train/mnt_train/process_data
python process_bags.py \
  --dataset-name husky \
  --input-dir /path/to/ros1_bags \
  --output-dir /path/to/processed_dataset \
  --sample-rate 4.0
```

### Where to specify which ROS topics to use

The bag-processing topic selection is configured in:

- `train/mnt_train/process_data/process_bags_config.yaml`

### If your bag format is different

If your camera, LiDAR, or odometry message format is different, add or adapt the processing functions in:

- `train/mnt_train/process_data/process_data_utils.py` for ROS1
- `train/mnt_train/process_data/process_data_utils_ros2.py` for ROS2

This is where image conversion, point-cloud parsing, and odometry-to-`(x, y, yaw)` conversion are defined.

### Recommended preprocessing flow

1. Run `process_bags.py` or `process_bags_ros2.py` to extract raw images, LiDAR frames, and `traj_data.pkl`.
2. Run `build_archives.py` on the generated trajectory folders to create `images.tar` and `points.zarr`.
3. Point the dataset entries in `train/config/vilint.yaml` to the resulting dataset root and splits.