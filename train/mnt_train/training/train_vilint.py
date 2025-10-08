from .train_utils import *
import torch.distributed as dist
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import FuncNorm

from torch.utils.tensorboard import SummaryWriter  

def _is_main_process(use_ddp: bool) -> bool:  
    return (not use_ddp) or (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0)

def unwrap_model(m):
    return m.module if hasattr(m, "module") else m

def train_eval_loop_vilint(
    train_model: bool,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: Adam, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    use_ddp: bool = False,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    use_tb: bool = True,
    tb_log_dir: Optional[str] = None,
    tb_writer: Optional[SummaryWriter] = None,
    pc_finetune_frac: float = 0.0, # 0.0 no training, 1.0 retrain allongside transformer
    pc_lr_scale: float = 0.1,
    image_mask_prob: float = 0.1,
    lidar_mask_prob: float = 0.0,
):
    """
    Train and evaluate the model for several epochs

    Args:
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler to use
        noise_scheduler: noise scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        goal_mask_prob: probability of masking the goal token during training
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images
        num_images_log: number of images to log
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
        pc_finetune_frac: fraction of epochs to fine-tune the pc encoder (unfreeze late in training)
        pc_lr_scale: learning rate scale for pc encoder params when unfrozen
    """
    latest_path = os.path.join(project_folder, f"latest.pth")

    created_writer_here = False
    if use_tb and tb_writer is None and _is_main_process(use_ddp):
        tb_writer = SummaryWriter(log_dir=tb_log_dir or os.path.join(project_folder, "tb"))
        created_writer_here = True

    # Optional fine-tuning of the MinkNet encoder
    total_epochs_in_call = epochs
    finetune_start_epoch = current_epoch + max(0, int((1.0 - pc_finetune_frac) * total_epochs_in_call))
    did_unfreeze_pc = False

    m = unwrap_model(model)

    for epoch in range(current_epoch, current_epoch + epochs):
        # Unfreeze pc encoder for late fine-tuning (if it was frozen at init)
        if (not did_unfreeze_pc
            and hasattr(m, 'vision_encoder')
            and getattr(m.vision_encoder, 'freeze_pc_encoder', False)
            and epoch >= finetune_start_epoch):
            try:
                m.vision_encoder.set_pc_encoder_trainable(True)
                # collect params that are not yet in the optimizer
                existing = set()
                for g in optimizer.param_groups:
                    for p in g['params']:
                        existing.add(id(p))
                new_params = [p for p in m.vision_encoder.pc_encoder.parameters() if id(p) not in existing]
                if len(new_params) > 0:
                    base_lr = optimizer.param_groups[0].get('lr', 1e-4)
                    optimizer.add_param_group({
                        'params': new_params,
                        'lr': base_lr * pc_lr_scale,
                    })
                print(f"[ViLiNT] Unfroze pc_encoder at epoch {epoch}; added {len(new_params)} params with lr scale {pc_lr_scale}.")
                did_unfreeze_pc = True
            except Exception as e:
                print(f"[ViLiNT] Failed to unfreeze pc_encoder: {e}")

        if train_model:
            print(
            f"Start ViLiNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            if use_ddp: train_loader.sampler.set_epoch(epoch)
            train_vilint(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                alpha=alpha,
                use_ddp=use_ddp,
                use_tb=use_tb,
                tb_writer=tb_writer,
                global_step_start=epoch * len(train_loader),
                image_mask_prob=image_mask_prob,
                lidar_mask_prob=lidar_mask_prob,
            )
            lr_scheduler.step()
        
        if (dist.get_rank() == 0 if use_ddp else True):
            numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
            torch.save(ema_model.averaged_model.state_dict(), numbered_path)
            numbered_path = os.path.join(project_folder, f"ema_latest.pth")
            torch.save(ema_model.averaged_model.state_dict(), numbered_path)
            print(f"Saved EMA model to {numbered_path}")

            numbered_path = os.path.join(project_folder, f"{epoch}.pth")
            torch.save(m.state_dict(), numbered_path)
            torch.save(m.state_dict(), latest_path)
            print(f"Saved model to {numbered_path}")

            # save optimizer
            numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
            latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
            torch.save(optimizer.state_dict(), latest_optimizer_path)

            # save scheduler
            numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
            latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
            torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

            if use_tb and tb_writer is not None:  # NEW
                tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        if use_ddp: dist.barrier()


        if (epoch + 1) % eval_freq == 0: 
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} ViLiNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                # Ensure EMA model uses the same LiDAR token source as the training model
                try:
                    if hasattr(ema_model, 'averaged_model') and hasattr(ema_model.averaged_model, 'vision_encoder') \
                    and hasattr(m, 'vision_encoder') and hasattr(m.vision_encoder, 'collision_lidar_source'):
                        ema_model.averaged_model.vision_encoder.set_collision_lidar_source(
                            m.vision_encoder.collision_lidar_source
                        )
                except Exception as e:
                    print(f"[ViLiNT] Warning: failed to sync EMA lidar source: {e}")
                if use_ddp: loader.sampler.set_epoch(epoch)
                evaluate_vilint(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    eval_fraction=eval_fraction,
                    use_ddp=use_ddp,
                    use_tb=use_tb,
                    tb_writer=tb_writer,
                )

        if use_ddp: dist.barrier()

    if created_writer_here and tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    print()



def train_vilint(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_ddp: bool = False,
    n_samples_per_condition: int = 20,
    use_tb: bool = True,
    tb_writer: Optional[SummaryWriter] = None,
    global_step_start: int = 0,
    image_mask_prob: float = 0.1,
    lidar_mask_prob: float = 0.0,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
    """
    global_step = global_step_start

    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    m = unwrap_model(model)
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", "train", window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )

    gc_action_loss_logger = Logger("gc_action_loss", "train", window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_col_loss_logger = Logger("uc_col_loss", "train", window_size=print_log_freq)
    gc_col_loss_logger = Logger("gc_col_loss", "train", window_size=print_log_freq)
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
        "uc_col_loss": uc_col_loss_logger,
        "gc_col_loss": gc_col_loss_logger,
    }   
    if use_ddp: dist.barrier()
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image,
                obs_lidar, 
                goal_coord,
                physics,
                actions,
                distance,
                dataset_idx,
                action_mask,
                collision_status, 
                col_mask,
                is_lidar,
            ) = data
            
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)

            lidar_mask = torch.zeros((obs_lidar.shape[0],), device=device).long()
            lidar_mask[is_lidar == 0] = 1  # mask out all LiDAR inputs for samples without LiDAR
            lidar_loss_mask = 1 - lidar_mask.float()

            if hasattr(m.vision_encoder, 'pc_encoder_type'):
                batch_obs_lidar = voxelize_pc_for_models(obs_lidar, device, model_type=m.vision_encoder.pc_encoder_type, lidar_mask=lidar_mask)
            else:
                batch_obs_lidar = None

            physics = physics.to(device)
            goal_coord = goal_coord.to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            # --- Modality masking ---
            # Randomly mask images (and optionally LiDAR) per sample.
            image_mask = (torch.rand((B,), device=device) < image_mask_prob).long()
            obs_feats = m("vision_encoder", obs_img=batch_obs_images, obs_pc=batch_obs_lidar, physics=physics, goal_coord=goal_coord,
                              input_goal_mask=goal_mask, input_image_mask=image_mask, input_lidar_mask=lidar_mask)
            scene = obs_feats["scene"]           # [B,D]
            lidar_ctx = obs_feats["lidar_ctx"]   # [B,K,D]
        
            B, D = scene.shape
            scene_mask = (torch.rand(B, device=device) < 0.10).unsqueeze(1)
            scene = torch.where(scene_mask, m.vision_encoder.empty_obs_token.expand(B, D).to(device), scene)

            deltas = get_delta(actions[...,:2])
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(np.concatenate((ndeltas, actions[...,2:]), axis=-1)).to(device)
            assert naction.shape[-1] == 2 or naction.shape[-1] ==  4, "action dim must be 2 or 4"

            # Predict distance
            dist_pred = m("dist_pred_net", obsgoal_cond=scene).to(device)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1).to(device), distance.float().to(device))
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 +(1 - goal_mask.float()).mean())

            logits = m("collision_pred_net",
                lidar_ctx=lidar_ctx,                 # [B,K,D]
                traj=actions[...,:2].to(device),     # [B,T,2|4]
                width=physics[:, :2],                # [B,2] -> (w,l)
                scene=scene)
            # Ensure logits are [B, T] in case the head returns [B, T, 1]
            if logits.dim() == 3 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)

            # Per-waypoint masked BCE over LiDAR ROI with dynamic class balancing
            targets = collision_status.to(device).to(dtype=logits.dtype)
            loss_col = asymmetric_huber_loss(logits, targets, w_over=5.0)
            
            def reduce_loss(unreduced_loss: torch.Tensor, mask: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == mask.shape, f"{unreduced_loss.shape} != {mask.shape}"
                return (unreduced_loss * mask).mean() / (mask.to(torch.float32).mean() + 1e-2)
            
            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_action = noise_scheduler.add_noise(
                naction, noise, timesteps)
            
            # Predict the noise residual
            noise_pred = m("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=scene)

            # L2 loss
            policy_loss = reduce_loss(F.mse_loss(noise_pred, noise, reduction="none"), action_mask)
            loss_col = reduce_loss(loss_col, lidar_loss_mask)

            beta_col = 0.3
            # Total loss
            if policy_loss != 0:
                loss = alpha * dist_loss + (1-alpha) * policy_loss + beta_col * loss_col
            else:
                loss = alpha * dist_loss + beta_col * loss_col
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(m)

            # Logging
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            if use_tb and tb_writer is not None and _is_main_process(use_ddp):
                if print_log_freq != 0 and (i % print_log_freq) == 0:
                    tb_writer.add_scalar("train/total_loss", float(loss_cpu), global_step)
                    tb_writer.add_scalar("train/dist_loss", float(dist_loss.item()), global_step)
                    tb_writer.add_scalar(
                        "train/policy_loss",
                        float(policy_loss.item()) if torch.is_tensor(policy_loss) else float(policy_loss),
                        global_step,
                    )


            if i % print_log_freq == 0:
                # Ensure EMA model uses the same LiDAR token source as the training model
                try:
                    if hasattr(ema_model, 'averaged_model') and hasattr(ema_model.averaged_model, 'vision_encoder') \
                       and hasattr(m, 'vision_encoder') and hasattr(m.vision_encoder, 'collision_lidar_source'):
                        ema_model.averaged_model.vision_encoder.set_collision_lidar_source(
                            m.vision_encoder.collision_lidar_source
                        )
                except Exception as e:
                    print(f"[ViLiNT] Warning: failed to sync EMA lidar source: {e}")
                losses = _compute_losses_vilint(
                            ema_model.averaged_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_obs_lidar,
                            physics,
                            goal_coord,
                            collision_status,
                            col_mask,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                            lidar_mask,
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())

                for key, logger in loggers.items():
                    if i % print_log_freq == 0 and print_log_freq != 0 and dist.get_rank() == 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
                        if use_tb and tb_writer is not None and _is_main_process(use_ddp):
                            tb_writer.add_scalar(f"train/{key}", float(logger.latest()), global_step)
                    if use_ddp: dist.barrier()
                    
            if image_log_freq != 0 and i % image_log_freq == 0 and (dist.get_rank()==0 if use_ddp else True):
                visualize_diffusion_action_distribution_vilint(
                    ema_model.averaged_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_viz_obs_images,
                    obs_lidar,
                    lidar_mask,
                    physics,
                    actions,
                    distance,
                    goal_coord,
                    collision_status,
                    device,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    B,
                    use_tb=use_tb,
                    tb_writer=tb_writer,
                    global_step=global_step,
                )
            torch.cuda.empty_cache()
            if use_ddp: dist.barrier()
    global_step += 1
    if use_ddp: dist.barrier()

def model_output_vilint(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_obs_lidar: torch.Tensor,
    lidar_mask: torch.Tensor,
    physics_tensor: torch.Tensor,
    goal_coord: torch.Tensor,
    actions: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
):
    # Build UC (goal masked) and GC (goal visible) encoder features
    goal_mask = torch.ones((goal_coord.shape[0],)).long().to(device)
    uc_feats = model("vision_encoder", obs_img=batch_obs_images, obs_pc=batch_obs_lidar,
                     physics=physics_tensor, goal_coord=goal_coord,
                     input_goal_mask=goal_mask, input_image_mask=None, input_lidar_mask=lidar_mask)
    uc_scene = uc_feats["scene"]            # [B,D]
    uc_lidar = uc_feats["lidar_ctx"]        # [B,K,D]
    uc_scene_rep = uc_scene.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((goal_coord.shape[0],)).long().to(device)
    gc_feats = model("vision_encoder", obs_img=batch_obs_images, obs_pc=batch_obs_lidar,
                     physics=physics_tensor, goal_coord=goal_coord,
                     input_goal_mask=no_mask, input_image_mask=None, input_lidar_mask=lidar_mask)
    gc_scene = gc_feats["scene"]            # [B,D]
    gc_lidar = gc_feats["lidar_ctx"]        # [B,K,D]
    gc_scene_rep = gc_scene.repeat_interleave(num_samples, dim=0)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(uc_scene_rep), pred_horizon, action_dim), device=device)

    # Diffusion UC
    diffusion_output = noisy_diffusion_output
    for k in noise_scheduler.timesteps[:]:
        noise_pred = model("noise_pred_net",
                            sample=diffusion_output,
                            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
                            global_cond=uc_scene_rep)
        diffusion_output = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=diffusion_output).prev_sample
    uc_actions = get_action(diffusion_output[:, :], ACTION_STATS)
    uc_col_pred = model("collision_pred_net", lidar_ctx=uc_lidar, traj=actions[...,:2].to(device),
                        width=physics_tensor[...,:2].to(device), scene=uc_scene)

    # Diffusion GC
    noisy_diffusion_output = torch.randn(
        (len(gc_scene_rep), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output
    for k in noise_scheduler.timesteps[:]:
        noise_pred = model("noise_pred_net",
                            sample=diffusion_output,
                            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
                            global_cond=gc_scene_rep)
        diffusion_output = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=diffusion_output).prev_sample
    gc_actions = get_action(diffusion_output[:, :], ACTION_STATS)
    gc_col_pred = model("collision_pred_net", lidar_ctx=gc_lidar, traj=actions[...,:2].to(device),
                        width=physics_tensor[...,:2].to(device), scene=gc_scene)

    # Distance head on GC scene
    gc_distance = model("dist_pred_net", obsgoal_cond=gc_scene)

    return {
        'uc_actions': uc_actions,
        'gc_actions': gc_actions,
        'gc_distance': gc_distance,
        'uc_col_pred': uc_col_pred,
        'gc_col_pred': gc_col_pred,
    }

def _compute_losses_vilint(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_obs_lidar,
    physics_tensor,
    goal_coord,
    col_status,
    col_mask,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
    lidar_mask: torch.Tensor,
):
    """
    Compute losses for distance and action prediction.
    """

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    model_output_dict = model_output_vilint(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_obs_lidar,
        lidar_mask,
        physics_tensor,
        goal_coord,
        batch_action_label,
        pred_horizon,
        action_dim,
        num_samples=1,
        device=device,
    )
    uc_actions = model_output_dict['uc_actions']
    gc_actions = model_output_dict['gc_actions']
    gc_distance = model_output_dict['gc_distance']
    uc_col_status = model_output_dict['uc_col_pred']
    gc_col_status = model_output_dict['gc_col_pred']

    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    def reduce_loss(unreduced_loss: torch.Tensor, mask: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == mask.shape, f"{unreduced_loss.shape} != {mask.shape}"
        return (unreduced_loss * mask).mean() / (mask.to(torch.float32).mean() + 1e-2)
    lidar_mask_loss = 1 - lidar_mask.float()
    uc_col_loss = reduce_loss(asymmetric_huber_loss(uc_col_status, col_status.to(device), w_over=5.0), lidar_mask_loss)
    gc_col_loss = reduce_loss(asymmetric_huber_loss(gc_col_status, col_status.to(device), w_over=5.0), lidar_mask_loss)
    
    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert uc_actions.shape == batch_action_label.shape, f"{uc_actions.shape} != {batch_action_label.shape}"
    assert gc_actions.shape == batch_action_label.shape, f"{gc_actions.shape} != {batch_action_label.shape}"

    uc_action_loss = reduce_loss(F.mse_loss(uc_actions, batch_action_label, reduction="none"), action_mask)
    gc_action_loss = reduce_loss(F.mse_loss(gc_actions, batch_action_label, reduction="none"), action_mask)

    uc_action_waypts_cos_similairity = reduce_loss(F.cosine_similarity(
        uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ), action_mask)
    uc_multi_action_waypts_cos_sim = reduce_loss(F.cosine_similarity(
        torch.flatten(uc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ), action_mask)

    gc_action_waypts_cos_similairity = reduce_loss(F.cosine_similarity(
        gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ), action_mask)
    gc_multi_action_waypts_cos_sim = reduce_loss(F.cosine_similarity(
        torch.flatten(gc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ), action_mask)

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
        "uc_col_loss": uc_col_loss,
        "gc_col_loss": gc_col_loss,
    }

    return results

def visualize_diffusion_action_distribution_vilint(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    obs_lidar: torch.Tensor,
    lidar_mask: torch.Tensor,
    physics_tensor: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    batch_col_status: torch.Tensor,
    device: torch.device,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    pc_scale: int = 5,
    use_tb: bool = True,
    tb_writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path, exist_ok=True)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]
    obs_lidar = obs_lidar[:num_images_log, :, :, :]
    if hasattr(ema_model.vision_encoder, 'pc_encoder_type'):
        batch_obs_lidar = voxelize_pc_for_models(obs_lidar, device, model_type=ema_model.vision_encoder.pc_encoder_type)
    else:
        batch_obs_lidar = None
    
    physics_tensor = physics_tensor[:num_images_log]
    batch_col_status = batch_col_status[:num_images_log]
    batch_robot_sizes = physics_tensor[:num_images_log]
    lidar_mask = lidar_mask[:num_images_log]

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []
    uc_col_list = []
    gc_col_list = []

    
    model_output_dict = model_output_vilint(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_obs_lidar,
        lidar_mask,
        physics_tensor,
        batch_goal_pos,
        batch_action_label,
        pred_horizon,
        action_dim,
        num_samples=10,
        device=device,
    )
    uc_actions_list.append(to_numpy(model_output_dict['uc_actions'][...,:2]))
    gc_actions_list.append(to_numpy(model_output_dict['gc_actions'][...,:2]))
    gc_distances_list.append(to_numpy(model_output_dict['gc_distance']))
    uc_col_list.append(to_numpy(model_output_dict['uc_col_pred']))
    gc_col_list.append(to_numpy(model_output_dict['gc_col_pred']))


    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)
    uc_col_list = np.concatenate(uc_col_list, axis=0)
    gc_col_list = np.concatenate(gc_col_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)
    uc_col_list  = np.split(uc_col_list, num_images_log, axis=0)
    gc_col_list  = np.split(gc_col_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i][...,:2])

        traj_list = np.concatenate([
            uc_actions,
            gc_actions,
            action_label[None],
        ], axis=0)
        # traj_labels = ["r", "GC", "GC_mean", "GT"]
        traj_colors = ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas, 
        )
        
        obs_image = to_numpy(batch_viz_obs_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)

        ax[1].imshow(obs_image)
        ax[2].imshow(obs_image)

        # set title
        ax[0].set_title(f"diffusion action predictions")
        ax[1].set_title(f"observation")
        ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        if use_tb and tb_writer is not None:
            tb_writer.add_figure(f"{eval_type}/action_samples/sample_{i}", fig, global_step if global_step is not None else epoch)
        plt.close(fig)

    for i in range(num_images_log):
        fig, ax = plt.subplots(figsize=(8,8))

        # Plot only one GC trajectory for the LiDAR point cloud figure.
        gc_i = gc_actions_list[i]
        if isinstance(gc_i, np.ndarray) and gc_i.ndim == 3:
            # shape (S, T, 2): take the first sampled trajectory
            gc_i = gc_i[0]
        traj_list = [to_numpy(batch_action_label[i][...,:2]), gc_i]
        
        col_probs_list = [to_numpy(batch_col_status[i])[:, np.newaxis], gc_col_list[i].T]

        colors_list = [plt.cm.Blues, plt.cm.Reds, plt.cm.Oranges]
        robot_size = to_numpy(batch_robot_sizes[i])
        markers = ['o'] * len(traj_list)
        linestyles = ['solid'] * len(traj_list)
        labels = [None] * len(traj_list)

        lidar = to_numpy(obs_lidar[i, 0, :, :])

        norm = mcolors.Normalize(vmin=lidar[:, 2].min(), vmax=lidar[:, 2].max())
        scatter = ax.scatter(lidar[:, 0], lidar[:, 1], c=lidar[:, 2], cmap='viridis', norm=norm, s=5, edgecolors='none',alpha=0.8, label='LiDAR Point Cloud')
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Z (meters)', rotation=270, labelpad=10)

        def forward(x):
            return  x
        def inverse(x):
            return x
        
        norm = FuncNorm((forward, inverse), vmin=0., vmax=1.0)

        for idx in range(len(traj_list)):
            traj = traj_list[idx]
            traj = np.concatenate((np.array([[0.0, 0.0]]), traj), axis=0)
            col_probs = col_probs_list[idx]
            col_probs = np.concatenate((col_probs, np.array([[0.0]]))).ravel()
            d = np.array(traj[1:] - traj[:-1])
            d = np.concatenate((d, d[np.newaxis, -1]))
            segments = np.stack([traj[:-1], traj[1:]], axis=1)

            cmap = colors_list[idx]
            label = labels[idx]
            ls = linestyles[idx]
            marker = markers[idx]

            # if idx == 0:
            scatter_pts = ax.scatter(
                traj[:,0],
                traj[:,1],
                c=col_probs,
                cmap=cmap,
                norm=norm,
                s=50,
                edgecolors='k',
                marker=marker,
                label=f'{label} (points)' if label else None
            )

            circle_edgecolor = 'black'
            circle_alpha = 0.1
            
            for (x, y, d0, d1, width) in zip(traj[..., 0], traj[..., 1], d[..., 0], d[..., 1], col_probs):
                angle = np.degrees(np.arctan2(d1, d0))
                width = float(np.max(robot_size[:2])*width)
                dist = np.sqrt(d0**2 + d1**2)
                if dist < 1e-4 or width < 1e-4:
                    continue
                rect = Rectangle(
                    (x , y-width/2.0),  # Lower-left corner
                    dist,  # Width of the rectangle
                    width,  # Height of the rectangle
                    rotation_point=(x,y), 
                    linewidth=1,
                    edgecolor=circle_edgecolor,
                    facecolor=circle_edgecolor,
                    alpha=circle_alpha,
                    angle=angle
                )
                ax.add_patch(rect) 

            seg_probs = 0.5 * (col_probs[:-1] + col_probs[1:])
            seg_probs = seg_probs.squeeze()

            lc = LineCollection(
                segments,
                array=seg_probs,
                cmap=cmap,
                norm=norm,
                linewidths=2,
                linestyles=ls
            )
            ax.add_collection(lc)

            cb_label = f'{label} Collision Probability (segments)' if label else 'Collision Probability (segments)'
            cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cb_label, rotation=270, labelpad=10)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('2D LiDAR Projection with Multiple Trajectories')
        ax.legend(loc='upper right')
        ax.set_aspect('equal', 'box')
        plt.tight_layout()

        save_path = os.path.join(visualize_path, f"lidar_proj_{i}.png")
        plt.savefig(save_path)
        if use_tb and tb_writer is not None:
            tb_writer.add_figure(f"{eval_type}/action_lidar_proj/sample_{i}", fig, global_step if global_step is not None else epoch)
        plt.close(fig)


def evaluate_vilint(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_ddp: bool = False,
    n_samples_per_condition: int = 20,
    use_tb: bool = True,
    tb_writer: Optional[SummaryWriter] = None,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs 
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model.averaged_model.eval()
    
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", eval_type, window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", eval_type, window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_col_loss_logger = Logger("uc_col_loss", eval_type, window_size=print_log_freq)
    gc_col_loss_logger = Logger("gc_col_loss", eval_type, window_size=print_log_freq)
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
        "uc_col_loss": uc_col_loss_logger,
        "gc_col_loss": gc_col_loss_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image,
                obs_lidar, 
                goal_coord,
                physics,
                actions,
                distance,
                dataset_idx,
                action_mask, 
                collision_status,
                col_mask,
                is_lidar,
            ) = data
            
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            N = batch_obs_images.shape[0]

            lidar_mask = torch.zeros((obs_lidar.shape[0],), device=device).long()
            lidar_mask[is_lidar == 0] = 1  # mask out all LiDAR inputs for samples without LiDAR
            
            if hasattr(ema_model.averaged_model.vision_encoder, 'pc_encoder_type'):
                batch_obs_lidar = voxelize_pc_for_models(obs_lidar, device, model_type=ema_model.averaged_model.vision_encoder.pc_encoder_type, lidar_mask=lidar_mask)
            else:
                batch_obs_lidar = None

            physics = physics.to(device)
            goal_coord = goal_coord.to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
            no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

            rand_mask_obs_feat = ema_model.averaged_model("vision_encoder", obs_img=batch_obs_images, obs_pc=batch_obs_lidar, physics=physics, goal_coord=goal_coord, input_goal_mask=rand_goal_mask, input_image_mask=None, input_lidar_mask=lidar_mask)
            rand_mask_scene = rand_mask_obs_feat["scene"]

            obs_feat = ema_model.averaged_model("vision_encoder", obs_img=batch_obs_images, obs_pc=batch_obs_lidar, physics=physics, goal_coord=goal_coord, input_goal_mask=no_mask, input_image_mask=None, input_lidar_mask=lidar_mask)
            obs_scene = obs_feat["scene"]
            obs_scene= obs_scene.flatten(start_dim=1)

            goal_mask_feat = ema_model.averaged_model("vision_encoder", obs_img=batch_obs_images, obs_pc=batch_obs_lidar, physics=physics, goal_coord=goal_coord, input_goal_mask=goal_mask, input_image_mask=None, input_lidar_mask=lidar_mask)
            goal_mask_scene = goal_mask_feat["scene"]
            distance = distance.to(device)

            deltas = get_delta(actions[..., :2])
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(np.concatenate((ndeltas, actions[...,2:]), axis=-1)).to(device)
            # collision_status = collision_status.unsqueeze(-1).to(device)
            assert naction.shape[-1] == 2 or naction.shape[-1] == 4, "action dim must be 2 or 4"

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)

            ### RANDOM MASK ERROR ###
            # Predict the noise residual
            rand_mask_noise_pred = ema_model.averaged_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=rand_mask_scene)
            
            # L2 loss
            rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, noise)
            
            ### NO MASK ERROR ###
            # Predict the noise residual
            no_mask_noise_pred = ema_model.averaged_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=obs_scene)
            
            # L2 loss
            no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, noise)

            ### GOAL MASK ERROR ###
            # predict the noise residual
            goal_mask_noise_pred = ema_model.averaged_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=goal_mask_scene)
            
            # L2 loss
            goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)
            
            # Logging
            loss_cpu = rand_mask_loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            
            if use_tb and tb_writer is not None and (dist.get_rank()==0 if use_ddp else True):
                if print_log_freq != 0 and (i % print_log_freq) == 0:
                    tb_writer.add_scalar(f"{eval_type}/diffusion_eval_loss_random_masking", float(rand_mask_loss.item()), epoch)
                    tb_writer.add_scalar(f"{eval_type}/diffusion_eval_loss_no_masking", float(no_mask_loss.item()), epoch)
                    tb_writer.add_scalar(f"{eval_type}/diffusion_eval_loss_goal_masking", float(goal_mask_loss.item()), epoch)
            
            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = _compute_losses_vilint(
                            ema_model.averaged_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_obs_lidar,
                            physics,
                            goal_coord,
                            collision_status,
                            col_mask,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                            lidar_mask,
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())

                for key, logger in loggers.items():
                    if i % print_log_freq == 0 and print_log_freq != 0 and (dist.get_rank()==0 if use_ddp else True):
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
                
                if use_tb and tb_writer is not None and (dist.get_rank()==0 if use_ddp else True):
                    for key, logger in loggers.items():
                        tb_writer.add_scalar(f"{eval_type}/{key}", float(logger.latest()), epoch)

            if image_log_freq != 0 and i % image_log_freq == 0 and (dist.get_rank()==0 if use_ddp else True):
                visualize_diffusion_action_distribution_vilint(
                    ema_model.averaged_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_viz_obs_images,
                    obs_lidar,
                    lidar_mask,
                    physics,
                    actions,
                    distance,
                    goal_coord,
                    collision_status,
                    device,
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    N*B,
                    use_tb=use_tb,
                    tb_writer=tb_writer,
                    global_step=None,
                )
            torch.cuda.empty_cache()
            if use_ddp: dist.barrier()
    if use_ddp: dist.barrier()