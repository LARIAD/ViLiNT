# Setup

This document is the install guide for running this repository, not for reproducing the full upstream `dune` or `PointTransformerV3` training stacks.

The default ViLiNT configuration in this repo uses:

- `DUNE` as the image encoder,
- `PointTransformerV3` as the LiDAR encoder,
- `diffusion_policy` for the diffusion head,
- the local `mnt_train` package for data loading and training.

## Recommended base stack

- NVIDIA GPU with a working CUDA driver
- CUDA 12.8 user-space packages
- Python 3.12

## 1. Clone the repo with submodules

```bash
git clone --recurse-submodules https://git.pole-recherche.fr/Robotique/multimodal-navigation-transformer.git
cd multimodal-navigation-transformer

# If you already cloned without submodules:
git submodule update --init --recursive
```

## 2. Create the conda environment

```bash
conda create -n vilint python=3.12 -y
conda activate vilint

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
## 3. Install DUNE dependencies
```bash
pip install -U torchfix timm 'huggingface_hub>=0.22' transformers==4.57.6 accelerate einops torchmetrics optuna tensorboard matplotlib pandas jaxtyping omegaconf ipython black flake8 pylint rich ipykernel
```

## 4. Install the PTv3 runtime dependencies

```bash
pip install -U h5py pyyaml sharedarray tensorboardx yapf addict scipy plyfile termcolor 
pip install -U --no-build-isolation torch-cluster torch-scatter torch-sparse
pip install -U torch-geometric
cd PointTransformerV3/Pointcept/libs/pointops
python setup.py install
cd ~/
pip install --no-cache-dir --no-build-isolation flash-attn
```
### Install tricky dependencies from source

```bash
mkdir torch_libs
cd ~/torch_libs/cumm
pip install -e .
cd ~/torch_libs/spconv
pip install --no-build-isolation -e .
```
## 5. ViLiNT dependencies

```bash
cd multimodal-navigation-transformer
pip install -e diffusion
pip install -e train
pip install -U tqdm git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git "opencv-python-headless==4.8.0.76" wandb prettytable efficientnet-pytorch warmup-scheduler huggingface-hub==0.36.0 diffusers==0.11.1 lmdb vit-pytorch positional-encodings zarr
```

## 5. Put the pretrained checkpoints in `train/weights`

Load the frozen weights in:

- `train/weights/dune_vitbase14_336.pth`
- `train/weights/point_transforer_v3_nuscenes_semseg.pth`
- optional: `train/weights/model_MinkUNeXt_refined.pth`

Create the directory if needed:

```bash
mkdir -p train/weights
```

### DUNE checkpoint

```bash
wget -O train/weights/dune_vitbase14_336.pth \
  https://download.europe.naverlabs.com/dune/dune_vitbase14_336.pth
```

### PTv3 checkpoint
```bash
  wget -O train/weights/point_transformer_v3_nuscenes_semseg.pth \
  https://huggingface.co/Pointcept/PointTransformerV3/resolve/main/nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth
```

## 6. Configure your dataset paths

Edit with your datasets paths for datas and splits:

- `train/config/vilint.yaml`

The training loader expects archive-based trajectories, not loose images only. Each trajectory folder should contain the assets referenced in the main `README.md`, typically:

- `images.tar`
- `points.zarr`
- `trajectory.zarr/positions.zarr`
- `trajectory.zarr/yaws.zarr`
- `width_curve.zarr`

Each split directory should contain a `traj_names.txt` with the trajectories name for train/test.


## 7. Optional: build datasets from ROS bags

The bag-processing scripts are separate from core training and need ROS Python packages.

### ROS1 preprocessing

Used by:

- `train/mnt_train/process_data/process_bags.py`
- `train/mnt_train/process_data/process_data_utils.py`

Extra requirements:

- ROS1 environment sourced
- `rosbag`
- `sensor_msgs`
- `tf2_msgs`
- `cv_bridge`
- `opencv-python`

### ROS2 preprocessing

Used by:

- `train/mnt_train/process_data/process_bags_ros2.py`
- `train/mnt_train/process_data/process_data_utils_ros2.py`

Extra requirements:

- ROS2 environment sourced
- `rclpy`
- `rosbag2_py`
- `sensor_msgs_py`
- `cv_bridge`

After raw extraction, build the archive format with:

```bash
cd train/mnt_train/process_data
python build_archives.py --root /path/to/dataset/datas/trajectories --overwrite
```

## 8. Optional: ROS2 deployment

For `deployment/src/deploy_vilint.py`, keep the training environment above and also source your ROS2 installation.

Extra deployment requirements:

- `rclpy`
- `sensor_msgs_py`
- `cv_bridge`
- `tf_transformations`

## Troubleshooting

- `AssertionError: Make sure flash_attn is installed.`
  Install `flash-attn`, or change the PTv3 constructor in `train/mnt_train/models/vilint/vilint.py` to disable flash attention.
- After installation of the packages in conda, it is possible that a numpy version mismatch     error happens. Just downgrade to `numpy==1.*`
- ROS import errors during bag processing or deployment
  Activate the conda env first, then source the matching ROS environment in the same shell.