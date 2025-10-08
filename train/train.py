import os
import shutil
import argparse
import numpy as np
import yaml
import time
import pdb
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

import torch.multiprocessing as mp
import torch.distributed as dist
import functools

train_dir = Path(__file__).resolve().parent
project_root = train_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
IMPORT YOUR MODEL HERE
"""
sys.path.insert(0, str(project_root / "diffusion"))
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from mnt_train.models.vilint.vilint import Lint_obs, Lint, DenseNetwork, CollisionScoringHeadSeq
from mnt_train.models.vilint.ema_vilint import ModelEMA
from mnt_train.training.train_vilint import train_eval_loop_vilint


from mnt_train.data.vilint_dataset import ViLiNT_Dataset

def vilint_worker(gpu, ngpus_per_node, args, train_func, config):
    """
    Worker function for distributed training.
    """
    args.gpu = gpu
    args.rank = 0 * ngpus_per_node + gpu
    # Initialize the default distributed process group
    # - world_size is the number of processes
    # - rank is the unique identifier for this process
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:23456",
        world_size=args.world_size,
        rank=args.rank,)
    
    # Set the CUDA device for this process
    print(f"[R{args.rank}] setting CUDA device {args.gpu}")
    torch.cuda.set_device(gpu)  
    print(f"[R{args.rank}] now on {torch.cuda.current_device()} "
          f"({torch.cuda.get_device_name()})")

    # # Load the data
    train_vilint_dataset = []
    test_vilint_dataloaders = {}
    
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                if config["model_type"] == "vilint":
                    vilint_dataset = ViLiNT_Dataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_size_li=config["context_size_li"],
                        is_lidar=data_config["lidar"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                        distance_type=config["distance_type"],
                    )
                if data_split_type == "train":
                    train_vilint_dataset.append(vilint_dataset)
                else:
                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_vilint_dataloaders:
                        test_vilint_dataloaders[dataset_type] = {}
                    test_vilint_dataloaders[dataset_type] = vilint_dataset

    # combine all the datasets from different robots
    train_vilint_dataset = ConcatDataset(train_vilint_dataset)

    sampler = torch.utils.data.DistributedSampler(
        train_vilint_dataset, shuffle=True, drop_last=True, num_replicas=args.world_size, rank=args.rank,
    ) 
    train_vilint_loader = DataLoader(
        train_vilint_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=max(1, config["num_workers"] // args.world_size),
        drop_last=True,
        persistent_workers=True,
        sampler=sampler,
        pin_memory=True,
        prefetch_factor=2,
    )
    for dataset_type, dataset in test_vilint_dataloaders.items():
        sampler_test = torch.utils.data.DistributedSampler(
            dataset, shuffle=True, drop_last=True, num_replicas=args.world_size, rank=args.rank,)
        test_vilint_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=0,
            drop_last=True,
            sampler=sampler_test,
        )

    # Instantiate the model inside the worker to avoid pickling issues
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
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    collision_head = CollisionScoringHeadSeq(
        d_model=config["encoding_size"],
        nheads=config["mha_num_attention_heads"],
        max_steps=config["len_traj_pred"],
        use_length_cond=config["use_length_cond"],
    )
    noise_pred_net = ConditionalUnet1D(
            input_dim=2+2*int(config["learn_angle"]),
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
    model = Lint(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,  
        collision_head=collision_head,
    )
    ema_model = ModelEMA(Lint(
        vision_encoder=Lint_obs(
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
        ),
        noise_pred_net=ConditionalUnet1D(
            input_dim=2+2*int(config["learn_angle"]),
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        ),
        dist_pred_net=DenseNetwork(embedding_dim=config["encoding_size"]),
        collision_head = CollisionScoringHeadSeq(
            d_model=config["encoding_size"],
            nheads=config["mha_num_attention_heads"],
            max_steps=config["len_traj_pred"],
            use_length_cond=config["use_length_cond"],
        ),
    ), power=0.75 )

    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path)
        model.load_state_dict(latest_checkpoint, strict=False)
        ema_model.averaged_model.load_state_dict(latest_checkpoint, strict=False)

    model.cuda(args.gpu)
    model.to(args.gpu)
    ema_model.averaged_model.cuda(args.gpu)
    ema_model.averaged_model.to(args.gpu)


    # Convert the model's batch norm layers to synchronized batch norm
    if config["pc_encoder"] == "minkunext":
        import MinkowskiEngine as ME
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Wrap the model in DistributedDataParallel. This synchronizes gradients during backward pass
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=False)
    # Specify that the model's graph is static to avoid confusion when a parameter is used more than once during backprop
    net._set_static_graph()

    # Define other required methods for training with the parallelized model
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(net.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")
    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in net.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )
    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

    if config["warmup"]:
        print("Using warmup scheduler")
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=config["warmup_epochs"],
            after_scheduler=scheduler,
        )

    # Call the preconfigured training function
    train_func(model=net, ema_model=ema_model, noise_scheduler=noise_scheduler,
                lr_scheduler=scheduler,
                optimizer=optimizer,
                train_loader=train_vilint_loader,
                test_dataloaders=test_vilint_dataloaders,
                use_ddp=True,
                device=args.gpu,
                use_tb=config["use_tb"],)
    dist.destroy_process_group()

def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    if config["model_type"] == "vilint":
                        dataset = ViLiNT_Dataset(
                            data_folder=data_config["data_folder"],
                            data_split_folder=data_config[data_split_type],
                            dataset_name=dataset_name,
                            image_size=config["image_size"],
                            waypoint_spacing=data_config["waypoint_spacing"],
                            min_dist_cat=config["distance"]["min_dist_cat"],
                            max_dist_cat=config["distance"]["max_dist_cat"],
                            min_action_distance=config["action"]["min_dist_cat"],
                            max_action_distance=config["action"]["max_dist_cat"],
                            negative_mining=data_config["negative_mining"],
                            len_traj_pred=config["len_traj_pred"],
                            learn_angle=config["learn_angle"],
                            context_size=config["context_size"],
                            context_size_li=config["context_size_li"],
                            is_lidar=data_config["lidar"],
                            context_type=config["context_type"],
                            end_slack=data_config["end_slack"],
                            goals_per_obs=data_config["goals_per_obs"],
                            normalize=config["normalize"],
                            goal_type=config["goal_type"],
                            distance_type=config["distance_type"],
                        )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}
                        test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    
    if config["model_type"] == "vilint":
        print("Using Lint")
    else:
        raise ValueError(f"Model {config['model']} not supported")


    if config["model_type"] == "vilint":
        current_epoch = 0
        args = argparse.Namespace()
        args.world_size = len(config["gpu_ids"])
        args.gpu = None
        args.rank = None
        num_devices = len(config["gpu_ids"])
        print("Num devices: ", num_devices)
        # Predefine function to call in the worker
        # TensorBoard logs mirror the run folder: logs/<project>/<run_name> -> tensorboard/<project>/<run_name>
        tb_log_dir = os.path.join("tensorboard", config["project_name"], config["run_name"])
        preconfigured_func = functools.partial(train_eval_loop_vilint,
            train_model=config["train"],
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
            tb_log_dir=tb_log_dir,
        )
        # Spawn multi-processing task 
        mp.spawn(vilint_worker, nprocs=num_devices, args=(num_devices, args, preconfigured_func, config), join=True)
    print("FINISHED TRAINING")


if __name__ == "__main__":
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<not-set>"))
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  → device {i}: {torch.cuda.get_device_name(i)}")
    
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vilint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = train_dir / config_path

    with open(train_dir / "config" / "defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )
    shutil.copy2(
        config_path,
        os.path.join(config["project_folder"], config_path.name),
    )

    print(config)
    start_time = time.time()
    main(config)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds")
