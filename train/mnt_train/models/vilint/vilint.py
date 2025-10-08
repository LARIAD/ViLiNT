import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from typing import List, Dict, Optional, Tuple, Callable
from mnt_train.models.vilint.self_attention import PositionalEncoding
from mnt_train.models.vilint.vilint_utils import replace_bn_with_gn

PROJECT_ROOT = Path(__file__).resolve().parents[4]

class Lint_obs(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        context_size_li: int = 0,
        im_encoder: Optional[str] = "dune",
        obs_encoding_size: Optional[int] = 512,
        pc_encoder: Optional[str] = None,
        use_physics_encoding: Optional[bool] = True,
        freeze_pc_encoder: Optional[bool] = True,
        pc_encoder_channels: Optional[int] = 3,
        physics_dim: Optional[int] = 5,
        goal_dim: Optional[int] = 2,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        lidar_sectors: Optional[int] = 8,
        lidar_rings: Optional[int] = 8,
        add_cartesian_lidar_pe: Optional[bool] = False,
        range_stat_mode: Optional[str] = "softmin",
        softmin_gamma: Optional[float] = 8.0,
        collision_lidar_source: Optional[str] = "fused",
    ) -> None:
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size
        self.physics_dim = physics_dim
        self.goal_dim = goal_dim
        self.use_physics_encoding = use_physics_encoding
        if pc_encoder is not None:
            self.pc_encoder_type = pc_encoder
            self.freeze_pc_encoder = freeze_pc_encoder
            self.lidar_context_size = context_size_li 
            self.pc_encoder_channels = pc_encoder_channels

            self.lidar_sectors = lidar_sectors
            assert collision_lidar_source in ("raw", "fused"), "collision_lidar_source must be 'raw' or 'fused'"
            self.collision_lidar_source = collision_lidar_source

        # Initialize the observation encoder
        if im_encoder.split("-")[0] == "efficientnet":
            from efficientnet_pytorch import EfficientNet
            self.im_encoder = EfficientNet.from_name(im_encoder, in_channels=3) # context
            self.im_encoder = replace_bn_with_gn(self.im_encoder)
            self.num_im_features = self.im_encoder._fc.in_features
            self.im_encoder_type = "efficientnet"
        elif im_encoder == "dune":
            print(PROJECT_ROOT)
            sys.path.insert(0, str(PROJECT_ROOT / "dune"))
            from model.dune import load_dune_encoder_from_checkpoint

            ckpt_path = PROJECT_ROOT / "train" / "weights" / "dune_vitbase14_336.pth"
            dune_obj = load_dune_encoder_from_checkpoint(ckpt_path)

            if isinstance(dune_obj, tuple):
                dune_obj = dune_obj[0]

            self.im_encoder = dune_obj

            self.num_im_features = 768
            self.im_encoder_type = "dune"

            for p in self.im_encoder.parameters():
                p.requires_grad = False
            self.im_encoder.eval()
        else:
            raise NotImplementedError

        # --- Point-cloud encoder selection ---
        if pc_encoder == ("pointtransformerv3"):
            sys.path.insert(0, str(PROJECT_ROOT / "PointTransformerV3"))
            from PointTransformerV3.model import PointTransformerV3
            self.pc_encoder = PointTransformerV3(in_channels=self.pc_encoder_channels, cls_mode=False, enable_flash=True)
            self.pc_encoder_out_channels = 64
            PATH = PROJECT_ROOT / "train" / "weights" / "point_transforer_v3_nuscenes_semseg.pth"
            ckpt = torch.load(PATH, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
            sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
            msg = self.pc_encoder.load_state_dict(sd, strict=False)
            self.pc_encoder.eval()
            print("[ViLiNT] Loaded PointTransformerV3 weights:", msg)
            print("using ", pc_encoder)
        elif pc_encoder == ("minkunext"):
            sys.path.insert(0, str(PROJECT_ROOT / "MinkUNeXt"))
            from MinkUNeXt.model.minkunext import MinkUNeXt, MinkUNeXtEncoder
            self.pc_encoder = MinkUNeXt(in_channels=self.pc_encoder_channels, out_channels=512, D=3, return_sparse=True)
            self.pc_encoder_out_channels = 512
            try:
                PATH = PROJECT_ROOT / "train" / "weights" / "model_MinkUNeXt_refined.pth"
                ckpt = torch.load(PATH, map_location="cpu", weights_only=False)
                sd = ckpt.get("state_dict", ckpt)
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                target = self.pc_encoder.state_dict()
                # Filter to matching shapes/keys to avoid errors
                loadable = {k: v for k, v in sd.items() if k in target and v.shape == target[k].shape}
                msg = self.pc_encoder.load_state_dict(loadable, strict=False)
                print("[ViLiNT] Loaded partial UNet weights:", msg)
                print("using ", pc_encoder)
            except Exception as e:
                print(f"[ViLiNT] Skipping UNet weight load: {e}")
        else:
            print(f"[ViLiNT] No point-cloud encoder selected (pc_encoder={pc_encoder})")

        if pc_encoder is not None:
            if self.obs_encoding_size != 512:
                self.compress_obs_li_enc = nn.Linear(512, self.obs_encoding_size)
            else:
                self.compress_obs_li_enc = nn.Identity()
            if freeze_pc_encoder:
                for param in self.pc_encoder.parameters():
                    param.requires_grad = False

            self.lidar_tokenizer = MinkNetTokenizer(
                in_dim=self.pc_encoder_out_channels,
                att_in_dim=self.obs_encoding_size,
                K=self.lidar_sectors,            # angular sectors (K_theta)
                radial_bins=lidar_rings,         # K_r rings
                use_gem=True,
                add_angle_pe=True,
                add_cartesian_pe=add_cartesian_lidar_pe,
                add_range_stats=True,
                range_stat_mode=range_stat_mode,
                softmin_gamma=softmin_gamma,
            )
            self.lidar_tokens = lidar_sectors * lidar_rings
        else:
            self.lidar_tokens = 0

        self.token_embeddings = ContextTokenEmbeddings(
            d_model=self.obs_encoding_size,
            img_history=self.context_size + 1,
            physics_in=self.physics_dim,
            goal_in=self.goal_dim,
            time_mode="learned",   # or "learned" if you prefer
            add_type=True,
        )

        # Initialize the goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(self.goal_dim, self.goal_encoding_size // 4),
            nn.ReLU(),
            nn.Linear(self.goal_encoding_size//4, self.goal_encoding_size//2),
            nn.ReLU(),
            nn.Linear(self.goal_encoding_size//2, self.goal_encoding_size)
        )
        if use_physics_encoding:
            self.physics_encoder = nn.Sequential(
                nn.Linear(self.physics_dim, self.obs_encoding_size // 4),
                nn.ReLU(),
                nn.Linear(self.obs_encoding_size//4, self.obs_encoding_size//2),
                nn.ReLU(),
                nn.Linear(self.obs_encoding_size//2, self.obs_encoding_size)
            )

        # Initialize compression layers if necessary
        if self.num_im_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_im_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        # Total number of tokens
        total_tokens = (self.context_size + 1) + self.lidar_tokens + (1 if self.use_physics_encoding else 0) + 1
        
        # Initialize positional encoding and self-attention layers
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=total_tokens)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

        # Learned attention pooling to produce a single scene token
        self.attn_pool = AttnPool1d(self.obs_encoding_size)

        # Learnt diffusion empty observation token
        self.empty_obs_token = nn.Parameter(torch.zeros(1, self.obs_encoding_size))

        # Base modality masks (broadcasted per batch in update_mask)
        self.goal_mask_base = torch.zeros((1, total_tokens), dtype=torch.bool)
        self.goal_mask_base[:, -1] = True
        self.image_mask_base = torch.zeros((1, total_tokens), dtype=torch.bool)
        self.image_mask_base[:, :self.context_size + 1] = True
        if pc_encoder is not None:
            self.lidar_mask_base = torch.zeros((1, total_tokens), dtype=torch.bool)
            self.lidar_mask_base[:, self.context_size + 1 : self.context_size + 1 + self.lidar_tokens] = True
        else:
            self.lidar_mask_base = None
        # Will be set per-forward when any mask is provided
        self.combined_mask = None
        self.avg_pool_mask = None


    def set_pc_encoder_trainable(self, trainable: bool) -> None:
        """Enable/disable training of the point-cloud encoder mid-training.
        This flips `requires_grad` on all pc_encoder params and sets module train/eval.
        """
        self.freeze_pc_encoder = not trainable
        for p in self.pc_encoder.parameters():
            p.requires_grad = trainable
        # put encoder in the appropriate mode (tokenizer etc. remain in model.train())
        self.pc_encoder.train(trainable)

    def forward(self, obs_img: torch.tensor, obs_pc: torch.tensor, physics_tensor: torch.tensor, goal_coord: torch.tensor, input_goal_mask: torch.tensor = None,
                input_image_mask: torch.tensor = None, input_lidar_mask: torch.tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        device = obs_img.device

        # Get the input goal mask 
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)
        
        # Goal token [B,1,d]
        goal_tok = self.token_embeddings.embed_goal(goal_coord)

        # Get the images encoding
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)
        if self.im_encoder_type == "efficientnet":
            im_encoding = self.im_encoder.extract_features(obs_img)
            im_encoding = self.im_encoder._avg_pooling(im_encoding)
            if self.im_encoder._global_params.include_top:
                im_encoding = im_encoding.flatten(start_dim=1)
                im_encoding = self.im_encoder._dropout(im_encoding)
        elif self.im_encoder_type == "dune":
            # DUNE encoder path
            with torch.no_grad():
                out = self.im_encoder(obs_img)
            if isinstance(out, dict):
                if "cls" in out:
                    im_encoding = out["cls"]                     # [B*T, D]
                else:
                    # fallback: first value in dict
                    im_encoding = next(iter(out.values()))
            elif isinstance(out, (list, tuple)):
                # take first element as features
                im_encoding = out[0]
            else:
                im_encoding = out

            # If we got a token sequence [B*T, N, D], reduce to a single vector
            if im_encoding.dim() == 3:
                # try CLS token if first position, else mean pool
                cls_candidate = im_encoding[:, 0]           # [B*T, D]
                if torch.isfinite(cls_candidate).all():
                    im_encoding = cls_candidate
                else:
                    im_encoding = im_encoding.mean(dim=1)   # [B*T, D]
        im_encoding = self.compress_obs_enc(im_encoding)
        im_encoding = im_encoding.unsqueeze(1)
        im_encoding = im_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        im_encoding = torch.transpose(im_encoding, 0, 1)
        # Add time embedding + modality embedding -> [B,T,d]
        t_idx = torch.arange(-self.context_size, 1, device=device)
        im_tokens = self.token_embeddings.add_time_and_type(im_encoding, t_idx, modality="image")
        
        if self.lidar_tokens > 0:
            # Get the LiDAR sector tokens [B,K,d]
            with torch.set_grad_enabled(not self.freeze_pc_encoder):
                enc = self.pc_encoder(obs_pc)
            
            # Build tokenizer input from encoder output
            if self.pc_encoder_type == "minkunet":
                if isinstance(enc, ME.SparseTensor):
                    enc_out = {
                        'feat': enc.F,
                        'coords': enc.C.int(),
                        'voxel_size': obs_pc.get('voxel_size', 1.0),
                        'tensor_stride': getattr(enc, 'tensor_stride', 1),
                    }
                else:
                    enc_out = {'global': enc}
            elif self.pc_encoder_type == "pointtransformerv3":
                enc_out = {
                    'feat': enc.feat,
                    'xyz': obs_pc['coord'],
                    'batch': obs_pc['batch'],
                }

            pc_tokens = self.lidar_tokenizer(enc_out)        # [B,K,d]
            chk("pc_tokens after tokenizer", pc_tokens)
            pc_tokens = self.token_embeddings.add_type(pc_tokens, modality="lidar")
            chk("pc_tokens after embeddings", pc_tokens)
            # Keep a copy of raw LiDAR tokens (pre-SA) for the collision head
            pc_tokens_raw = pc_tokens
            if input_lidar_mask is not None:
                # Zero-out raw tokens when LiDAR is masked for this sample
                lm_scale = (1 - input_lidar_mask.view(-1, 1, 1).float()).to(pc_tokens_raw.device)
                pc_tokens_raw = pc_tokens_raw * lm_scale
        else:
            pc_tokens = None
            pc_tokens_raw = None

        # Physics single token [B,1,d]
        if self.use_physics_encoding:
            phys_tok = self.token_embeddings.embed_physics(physics_tensor)

        # Final sequence: [B, T + K + (1 if physics) + 1, d]
        if pc_tokens is not None:
            parts = [im_tokens, pc_tokens]
        else:
            parts = [im_tokens]
        if self.use_physics_encoding:
            parts.append(phys_tok)
        parts.append(goal_tok)
        tokens = torch.cat(parts, dim=1)
        
        # Build combined modality padding mask if any was provided
        if (goal_mask is not None) or (input_image_mask is not None) or (input_lidar_mask is not None):
            self.update_mask(
                image_mask=input_image_mask.to(device) if input_image_mask is not None else None,
                lidar_mask=input_lidar_mask.to(device) if input_lidar_mask is not None else None,
                goal_mask=goal_mask.to(device) if goal_mask is not None else None,
            )
            src_key_padding_mask = self.combined_mask.to(device)
        else:
            src_key_padding_mask = None

        chk("tokens_after_PE", tokens)

        encoding_tokens = self.sa_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        # after transformer:
        chk("encoding_tokens", encoding_tokens)

        if self.lidar_tokens > 0:
            # Slice LiDAR tokens from fused sequence
            pc_start = self.context_size + 1
            pc_end = pc_start + self.lidar_tokenizer.K
            lidar_ctx_fused = encoding_tokens[:, pc_start:pc_end, :]  # [B,K,D]

            # Select raw (pre-SA) or fused tokens for the collision head
            if self.collision_lidar_source == "raw":
                lidar_ctx = pc_tokens_raw
            else:
                lidar_ctx = lidar_ctx_fused
        else:
            lidar_ctx = encoding_tokens[:, :self.context_size+2, :]
            lidar_ctx_fused = encoding_tokens[:, :self.context_size+2, :]

        # Scene token via learned attention pooling (respects mask)
        scene = self.attn_pool(encoding_tokens, key_padding_mask=src_key_padding_mask)  # [B,D]
        assert torch.isfinite(scene).all().item(), "NaN/Inf in pooled scene"

        return {"scene": scene, "lidar_ctx": lidar_ctx, "lidar_ctx_raw": pc_tokens_raw, "lidar_ctx_fused": lidar_ctx_fused}
    
    def update_mask(self, image_mask=None, lidar_mask=None, goal_mask=None):
        """
        Build a per-batch src_key_padding_mask (True = pad/mask) and the corresponding
        avg_pool_mask used to fairly average only the unmasked tokens.

        image_mask/lidar_mask/goal_mask are shape [B] with 1 to mask, 0 to keep.
        """
        # Determine batch size
        batch_size = 1
        for m in (goal_mask, image_mask, lidar_mask):
            if m is not None:
                batch_size = int(m.shape[0])
                break

        total_tokens = (self.context_size + 1) + self.lidar_tokens + (1 if self.use_physics_encoding else 0) + 1
        device = self.goal_mask_base.device

        combined = torch.zeros((batch_size, total_tokens), dtype=torch.bool, device=device)

        if goal_mask is not None:
            gm = goal_mask.view(batch_size, 1).bool().to(device)
            base = self.goal_mask_base.expand(batch_size, total_tokens)
            combined = combined | (gm & base)

        if image_mask is not None:
            im = image_mask.view(batch_size, 1).bool().to(device)
            base = self.image_mask_base.expand(batch_size, total_tokens)
            combined = combined | (im & base)

        if lidar_mask is not None and hasattr(self, 'pc_encoder'):
            lm = lidar_mask.view(batch_size, 1).bool().to(device)
            base = self.lidar_mask_base.expand(batch_size, total_tokens)
            combined = combined | (lm & base)

        # Save combined mask
        self.combined_mask = combined

        # Averaging weights: ignore masked tokens and preserve mean magnitude
        mask_float = combined.float()
        unmasked = 1.0 - mask_float
        keep = unmasked.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = unmasked * (total_tokens / keep)
        self.avg_pool_mask = weights
    
    def get_combined_mask(self):
        return self.combined_mask

    def set_collision_lidar_source(self, source: str):
        assert source in ("raw", "fused"), "source must be 'raw' or 'fused'"
        self.collision_lidar_source = source

class Lint(nn.Module):

    def __init__(self, vision_encoder, 
                       noise_pred_net,
                       dist_pred_net,
                       collision_head: Optional[nn.Module] = None):
        super(Lint, self).__init__()


        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
        self.collision_head = collision_head

    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder" :
            output = self.vision_encoder(kwargs["obs_img"], kwargs["obs_pc"], kwargs["physics"], kwargs["goal_coord"], input_goal_mask=kwargs["input_goal_mask"], input_image_mask=kwargs["input_image_mask"], input_lidar_mask=kwargs["input_lidar_mask"])
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        elif func_name == "collision_pred_net":
            output = self.collision_head(
                lidar_ctx=kwargs["lidar_ctx"],
                traj=kwargs["traj"],
                width=kwargs.get("width", None),
                scene=kwargs.get("scene", None),
            )
        else:
            raise NotImplementedError
        return output

class Lint_imle(nn.Module):
    
    def __init__(self, vision_encoder, policy_net, dist_pred_net,):
        super(Lint_imle, self).__init__()

        self.vision_encoder = vision_encoder
        self.policy_net = policy_net
        self.dist_pred_net = dist_pred_net
    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder" :
            output = self.vision_encoder(kwargs["obs_img"], kwargs["obs_pc"], kwargs["physics"], kwargs["goal_coord"], input_goal_mask=kwargs["input_goal_mask"], input_image_mask=kwargs["input_image_mask"], input_lidar_mask=kwargs["input_lidar_mask"])
        elif func_name == "policy_net":
            output = self.policy_net(global_cond=kwargs["global_cond"], sample=kwargs["sample"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output

class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim, use_sigmoid=False):
        super(DenseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim 
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 1)
        )
        if use_sigmoid:
            self.use_sigmoid = True
            self.sgm = nn.Sigmoid()
        else:
            self.use_sigmoid = False
    
    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        if self.use_sigmoid:
            output = self.sgm(output)
        return output
    
class MinkNetTokenizer(nn.Module):
    """
    Polar-grid (theta × r) sector-pooling LiDAR tokenizer with optional range-aware statistics.
    - Fixed token budget: K_total = K_theta * K_r
    - Optional GeM pooling on encoder features
    - Positional encodings: angle (sin/cos of sector center) and ring center range
    - Optional Cartesian stats (x̄,ȳ,r̄,logcount) — disabled by default
    """
    def __init__(
        self,
        in_dim: int,
        att_in_dim: int,
        K: int = 24,                 # angular sectors (K_theta)
        radial_bins: int = 1,        # radial rings (K_r)
        radial_edges: Optional[List[float]] = None,  # if None, uniform in [0, range_scale]
        fov_half_deg: float = 90.0,
        use_gem: bool = False,
        add_angle_pe: bool = True,
        add_cartesian_pe: bool = False,
        add_range_stats: bool = True,
        range_stat_mode: str = "softmin",  # {'softmin','harmonic','none'}
        softmin_gamma: float = 8.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        # geometry setup
        self.Ktheta = K
        self.Kr = max(1, int(radial_bins))
        self.K = self.Ktheta * self.Kr          # total tokens exposed to the rest of the model
        self.eps = eps
        self.use_gem = use_gem
        self.fov_half_rad = math.radians(fov_half_deg)
        self.add_angle_pe = add_angle_pe
        self.add_cartesian_pe = add_cartesian_pe
        self.add_range_stats = add_range_stats
        self.range_stat_mode = range_stat_mode
        self.softmin_gamma = softmin_gamma
        # range normalization scale (meters)
        self.range_scale = 10.0

        # projection to transformer size d_model
        self.proj = nn.Sequential(
            nn.Linear(in_dim, att_in_dim),
            nn.LayerNorm(att_in_dim),
            nn.GELU(),
        )

        # GeM pooling
        if use_gem:
            self.p = nn.Parameter(torch.ones(1) * 3.0)

        # For empty sectors
        self.empty = nn.Parameter(torch.zeros(1, 1, in_dim))

        # angle PE: project sin/cos(theta_center) to d_model and add
        if self.add_angle_pe:
            self.angle_proj = nn.Linear(2, att_in_dim)

        # ring PE: project normalized ring-center radius to d_model and add
        self.ring_proj = nn.Linear(1, att_in_dim)

        # Cartesian stats PE
        if self.add_cartesian_pe:
            # features: [x̄/R, ȳ/R, r̄/R, log1p(count)/10] (NOTE: no duplicate sin/cos here)
            self.xy_proj = nn.Linear(4, att_in_dim)

        # Range-aware stats PE (mean r and a near-biased statistic)
        if self.add_range_stats and self.range_stat_mode != "none":
            # [mean_r/R, near_r/R] → d_model
            self.range_stats_proj = nn.Linear(2, att_in_dim)
        else:
            self.range_stats_proj = None

        self.out_ln = nn.LayerNorm(att_in_dim)

        # optionally store explicit radial edges
        if radial_edges is not None:
            # Expect a Python list/tuple; we store as buffer for device moves
            edges = torch.tensor(radial_edges, dtype=torch.float32)
            assert edges.ndim == 1 and edges.numel() == self.Kr + 1, "radial_edges must have Kr+1 values"
            self.register_buffer("radial_edges", edges, persistent=False)
        else:
            self.register_buffer("radial_edges", None, persistent=False)

    @staticmethod
    def _coords_to_xyz(enc_out):
        """
        Approximate metric xyz from Mink integer coords
        coords = [b, x, y, z]; uses voxel_size * tensor_stride.
        """
        coords = enc_out['coords']  # [N, 4]
        voxel_size = enc_out['voxel_size']      # float
        tstride = enc_out.get('tensor_stride', 1)
        if isinstance(tstride, (list, tuple)):
            sx, sy, sz = tstride
        else:
            sx = sy = sz = tstride
        ijk = coords[:, 1:4].float()
        xyz = torch.stack([
            (ijk[:, 0] + 0.5) * (sx * voxel_size),
            (ijk[:, 1] + 0.5) * (sy * voxel_size),
            (ijk[:, 2] + 0.5) * (sz * voxel_size),
        ], dim=-1)
        batch = coords[:, 0].long()
        return xyz, batch

    def _sector_pool(self, xyz, feat, batch):
        """
        xyz:   [N, 3]  (meters, robot frame)
        feat:  [N, C]
        batch: [N]
        → pooled: [B, Kr, Ktheta, C], sums & counts for stats
        """
        device = feat.device
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        Ktheta = self.Ktheta
        Kr = self.Kr
        C = feat.size(1)

        # --- angular bins ---
        edges_theta = torch.linspace(-self.fov_half_rad, self.fov_half_rad, Ktheta + 1, device=device)
        theta = torch.atan2(xyz[:, 1], xyz[:, 0])                         # [-pi, pi]
        theta = torch.clamp(theta, -self.fov_half_rad, self.fov_half_rad)
        sec = torch.clamp(torch.bucketize(theta, edges_theta) - 1, 0, Ktheta - 1)  # [N]

        # --- radial bins ---
        r = torch.norm(xyz[:, :2], dim=-1, keepdim=True).clamp_min(self.eps)  # [N,1]
        if self.radial_edges is None:
            edges_r = torch.linspace(0.0, self.range_scale, Kr + 1, device=device)
        else:
            edges_r = self.radial_edges.to(device)
        ring = torch.clamp(torch.bucketize(r.squeeze(-1), edges_r) - 1, 0, Kr - 1)  # [N]

        # flat index per (b, ring, sector)
        flat = (batch * (Kr * Ktheta)) + (ring * Ktheta) + sec  # [N]

        pooled = feat.new_zeros((B * Kr * Ktheta, C))
        counts = feat.new_zeros((B * Kr * Ktheta, 1))

        if self.use_gem:
            p = torch.clamp(self.p, min=1.0)
            x_mag_p = torch.pow(feat.abs() + self.eps, p)
            pooled = pooled.index_add(0, flat, x_mag_p)
            counts = counts.index_add(0, flat, torch.ones_like(r))
            pooled = pooled / torch.clamp(counts, min=1.0)
            pooled = torch.pow(torch.clamp(pooled, min=0.0), 1.0 / p)
        else:
            pooled = pooled.index_add(0, flat, feat)
            counts = counts.index_add(0, flat, torch.ones_like(r))
            pooled = pooled / torch.clamp(counts, min=1.0)

        # Stats for PE
        sum_xy = xyz.new_zeros((B * Kr * Ktheta, 2))
        sum_r  = xyz.new_zeros((B * Kr * Ktheta, 1))
        sum_inv_r = xyz.new_zeros((B * Kr * Ktheta, 1))
        sum_exp_neg = xyz.new_zeros((B * Kr * Ktheta, 1))

        xy = xyz[:, :2]
        inv_r = 1.0 / (r + self.eps)
        exp_neg = torch.exp(-self.softmin_gamma * r)

        sum_xy = sum_xy.index_add(0, flat, xy)
        sum_r  = sum_r.index_add(0, flat, r)
        sum_inv_r = sum_inv_r.index_add(0, flat, inv_r)
        sum_exp_neg = sum_exp_neg.index_add(0, flat, exp_neg)

        # reshape to [B, Kr, Ktheta, *]
        pooled = pooled.view(B, Kr, Ktheta, C)
        counts = counts.view(B, Kr, Ktheta, 1)
        sum_xy = sum_xy.view(B, Kr, Ktheta, 2)
        sum_r  = sum_r.view(B, Kr, Ktheta, 1)
        sum_inv_r = sum_inv_r.view(B, Kr, Ktheta, 1)
        sum_exp_neg = sum_exp_neg.view(B, Kr, Ktheta, 1)

        return pooled, edges_theta, edges_r, sum_xy, sum_r, counts, sum_inv_r, sum_exp_neg

    def forward(self, enc_out: dict) -> torch.Tensor:
        # --- inputs ---
        if 'feat' in enc_out and 'xyz' in enc_out and 'batch' in enc_out:
            feat  = enc_out['feat']   # [N, C_in]
            xyz   = enc_out['xyz']    # [N, 3]
            batch = enc_out['batch']  # [N]
        elif 'feat' in enc_out and 'coords' in enc_out and 'voxel_size' in enc_out:
            feat  = enc_out['feat']
            xyz, batch = self._coords_to_xyz(enc_out)
        elif 'global' in enc_out:
            vec = enc_out['global']            # [B*CW, C_in]
            token = self.proj(vec)             # [B*CW, att_in_dim]
            return token
        else:
            raise ValueError(
                "Expected enc_out to contain ('feat','xyz','batch') or ('feat','coords','voxel_size'[,'tensor_stride']), or legacy 'global'."
            )

        # --- sector pooling on theta×r grid ---
        pooled, edges_theta, edges_r, sum_xy, sum_r, counts, sum_inv_r, sum_exp_neg = self._sector_pool(xyz, feat, batch)
        B, Kr, Ktheta, Cin = pooled.shape

        # project to transformer dim → [B, Kr, Ktheta, d]
        tokens = self.proj(pooled.view(B * Kr * Ktheta, Cin)).view(B, Kr, Ktheta, -1)

        # angle PE (broadcast over rings)
        if self.add_angle_pe:
            centers_theta = 0.5 * (edges_theta[:-1] + edges_theta[1:])              # [Ktheta]
            ang = torch.stack([torch.sin(centers_theta), torch.cos(centers_theta)], dim=-1)  # [Ktheta,2]
            ang = self.angle_proj(ang).view(1, 1, Ktheta, -1)               # [1,1,Ktheta,d]
            tokens = tokens + ang                                       # broadcast over B,Kr

        # ring PE (broadcast over sectors)
        centers_r = 0.5 * (edges_r[:-1] + edges_r[1:])                  # [Kr]
        ring_norm = (centers_r / self.range_scale).view(1, Kr, 1, 1)    # [1,Kr,1,1]
        ring_pe = self.ring_proj(ring_norm.view(1, Kr, 1, 1))           # linear works elementwise
        ring_pe = ring_pe.expand(B, Kr, Ktheta, -1)
        tokens = tokens + ring_pe

        # Optional Cartesian PE (no duplicate sin/cos)
        if self.add_cartesian_pe:
            counts_clamped = counts.clamp_min(1.0)
            mean_xy = sum_xy / counts_clamped                  # [B,Kr,Ktheta,2]
            mean_r  = sum_r  / counts_clamped                  # [B,Kr,Ktheta,1]
            logcnt  = torch.log1p(counts_clamped) / 10.0       # [B,Kr,Ktheta,1]
            feat_xy = torch.cat([
                mean_xy / self.range_scale,    # 2
                mean_r  / self.range_scale,    # 1
                logcnt                          # 1
            ], dim=-1)                           # -> 4 dims
            tokens = tokens + self.xy_proj(feat_xy)

        # Range-aware stats (mean and near-biased)
        if self.range_stats_proj is not None:
            counts_clamped = counts.clamp_min(1.0)
            mean_r = (sum_r / counts_clamped)                     # [B,Kr,Ktheta,1]
            if self.range_stat_mode == "harmonic":
                near = (counts_clamped / (sum_inv_r + self.eps))  # harmonic mean
            else:  # softmin (default)
                near = -torch.log((sum_exp_neg / counts_clamped).clamp_min(self.eps)) / self.softmin_gamma
            stats = torch.cat([
                (mean_r / self.range_scale),
                (near   / self.range_scale),
            ], dim=-1)                                            # [B,Kr,Ktheta,2]
            tokens = tokens + self.range_stats_proj(stats)

        # flatten rings × sectors → K_total
        tokens = tokens.view(B, self.K, -1)
        tokens = self.out_ln(tokens)
        return tokens

class ContextTokenEmbeddings(nn.Module):
    """
    - Time embeddings for sequence tokens (image or lidar).
    - Physics and goal tokens via small MLPs to d_model.
    - Optional type embeddings for each modality.
    """
    def __init__(
        self,
        d_model: int = 512,
        img_history: int = 4,
        physics_in: int = 3,
        goal_in: int = 2,
        time_mode: str = "sinusoid",
        add_type: bool = True,
        type_vocab=("image", "lidar", "physics", "goal"),
        dropout: float = 0.0,
    ):
        super().__init__()
        assert time_mode in {"learned", "sinusoid"}
        self.d_model = d_model
        self.img_history = img_history
        self.time_mode = time_mode
        self.add_type_flag = add_type

        # Map lags in [-img_history+1, ..., 0] to indices [0..img_history-1]
        self.time_offset = img_history - 1

        if time_mode == "learned":
            self.time_embed = nn.Embedding(img_history, d_model)
            nn.init.normal_(self.time_embed.weight, std=0.02)
        else:
            self.register_buffer("sin_base", self._make_sin_base(d_model), persistent=False)

        # Type embeddings (learned)
        if add_type:
            self.type_embed = nn.ParameterDict({
                name: nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
                for name in type_vocab
            })
        else:
            self.type_embed = None

        # Physics & goal MLPs (Linear -> LN -> GELU)
        self.phys_mlp = nn.Sequential(
            nn.Linear(physics_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.goal_mlp = nn.Sequential(
            nn.Linear(goal_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.out_ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    # public 
    
    def add_time_and_type(self, tokens: torch.Tensor, t_idx: torch.Tensor, modality: str) -> torch.Tensor:
        """
        tokens: [B, T, d]
        t_idx:  shape [T] or [B,T], values in [-img_history+1 .. 0]
        modality: "image" or "lidar" (or any in type_vocab)
        """
        B, T, D = tokens.shape
        assert D == self.d_model, f"d_model mismatch: got {D}, expected {self.d_model}"

        pe = self.time_embedding(t_idx, batch=B, T=T, device=tokens.device)  # [B,T,d_model]
        out = tokens + pe

        if self.add_type_flag:
            out = out + self.type_embed[modality]  # broadcast [1,1,d]

        out = self.out_ln(out)
        return self.drop(out)

    def add_type(self, tokens: torch.Tensor, modality: str) -> torch.Tensor:
        """
        tokens: [B, N, d]
        Adds only the modality/type embedding (no time).
        """
        if not self.add_type_flag:
            return tokens
        out = tokens + self.type_embed[modality]
        return self.out_ln(out)

    def embed_physics(self, phys_vec: torch.Tensor) -> torch.Tensor:
        """
        phys_vec: [B, physics_in]
        returns:  [B, 1, d_model]
        """
        tok = self.phys_mlp(phys_vec).unsqueeze(1)   # [B,1,d]
        if self.add_type_flag:
            tok = tok + self.type_embed["physics"]
        tok = self.out_ln(tok)
        return self.drop(tok)

    def embed_goal(self, goal_vec: torch.Tensor) -> torch.Tensor:
        """
        goal_vec: [B, goal_in]
        returns:  [B, 1, d_model]
        """
        tok = self.goal_mlp(goal_vec).unsqueeze(1)   # [B,1,d]
        if self.add_type_flag:
            tok = tok + self.type_embed["goal"]
        tok = self.out_ln(tok)
        return self.drop(tok)

    # private

    def time_embedding(self, t_idx: torch.Tensor, batch: int, T: int, device: torch.device) -> torch.Tensor:
        """
        t_idx: [T] or [B,T], integer lags in [-H+1..0]
        returns: [B, T, d_model]
        """
        if t_idx.dim() == 1:
            t_idx = t_idx.view(1, T).expand(batch, T)
        else:
            assert t_idx.shape[0] in (1, batch)
            if t_idx.shape[0] == 1:
                t_idx = t_idx.expand(batch, -1)

        # Map to [0 .. H-1]
        idx = t_idx + self.time_offset
        idx = idx.clamp_(0, self.img_history - 1).to(device)

        if self.time_mode == "learned":
            pe = self.time_embed(idx)             # [B,T,d]
        else:
            pe = self._sinusoid(idx)              # [B,T,d]

        return pe

    def _make_sin_base(self, d_model: int) -> torch.Tensor:
        """ [d_model//2] inverse frequencies for sinusoidal PE. """
        half = d_model // 2
        inv_freq = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=torch.float32) / half)
        return inv_freq  # [half]

    def _sinusoid(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: [B,T] in [0..H-1] -> sinusoidal PE [B,T,d_model]
        """
        B, T = idx.shape
        # Use idx as positions; can also remap to actual time gaps if unequal spacing.
        pos = idx.float().unsqueeze(-1)                          # [B,T,1]
        freqs = self.sin_base.view(1, 1, -1).to(idx.device)      # [1,1,half]
        angles = pos * freqs                                     # [B,T,half]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        pe = torch.zeros(B, T, self.d_model, device=idx.device)
        pe[..., 0::2] = sin
        pe[..., 1::2] = cos
        return pe

class CollisionScoringHeadSeq(nn.Module):
    """
    Per-waypoint collision scorer using cross-attention:
      - Queries: trajectory waypoints (x,y) conditioned on robot size (w,l)
      - Keys/Values: post-fusion LiDAR context tokens [B, K, D]
      - Optional scene FiLM
    Outputs: logits per waypoint [B, T]
    """
    def __init__(
        self,
        d_model: Optional[int] = None,
        embedding_dim: Optional[int] = None,  # legacy name
        nheads: int = 4,
        nlayers: int = 1,
        ffn_dim_factor: int = 4,
        max_steps: int = 128,
        use_scene_film: bool = True,
        use_length_cond: bool = False,
    ):
        super().__init__()
        d_model = d_model or embedding_dim or 512
        self.d_model = d_model
        self.use_scene_film = use_scene_film
        self.geom_cond_dim = 1 + use_length_cond

        # Encode [x,y,w,l] (+ step PE) -> d_model
        self.q_in = nn.Sequential(
            nn.Linear(2 + self.geom_cond_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.step_embed = nn.Embedding(max_steps, d_model)
        nn.init.normal_(self.step_embed.weight, std=0.02)

        self.cross_attn = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim_factor * d_model),
            nn.GELU(),
            nn.Linear(ffn_dim_factor * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        if self.use_scene_film:
            self.film_gamma = nn.Linear(d_model, d_model)
            self.film_beta = nn.Linear(d_model, d_model)

        self.readout = nn.Linear(d_model, 1)

    def forward(
        self,
        *,
        lidar_ctx: torch.Tensor,   # [B, K, D]
        traj: torch.Tensor,        # [B, T, 2]
        width: torch.Tensor,       # [B, 2] -> (w, l)
        scene: Optional[torch.Tensor] = None,  # [B, D]
    ) -> torch.Tensor:
        B, T, _ = traj.shape

        # Build queries from (x,y) + robot size per step
        if self.geom_cond_dim == 1:
            width = width[..., 0]
        if width.dim() == 1:
            width = width.unsqueeze(1)
        size_rep = width.unsqueeze(1).expand(B, T, self.geom_cond_dim)
        q_in = torch.cat([traj, size_rep], dim=-1)                    # [B,T,4]
        q = self.q_in(q_in)                                           # [B,T,D]

        # Add step index embedding
        t_idx = torch.arange(T, device=traj.device).view(1, T)
        q = q + self.step_embed(t_idx).expand(B, T, -1)

        # Cross-attend to LiDAR tokens
        attn_out, _ = self.cross_attn(q, lidar_ctx, lidar_ctx, need_weights=False)  # [B,T,D]
        h = self.norm1(q + attn_out)

        # Optional FiLM from scene token
        if (scene is not None) and self.use_scene_film:
            gamma = self.film_gamma(scene).unsqueeze(1)               # [B,1,D]
            beta = self.film_beta(scene).unsqueeze(1)                 # [B,1,D]
            h = gamma * h + beta

        # FFN
        h2 = self.ffn(h)
        h = self.norm2(h + h2)

        # Per-step logits
        logits = self.readout(h).squeeze(-1)                          # [B,T]
        return logits

class AttnPool1d(nn.Module):
    """Single-query attention pooling over a token sequence with optional mask.
    Returns [B, D].
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # tokens: [B, S, D]; key_padding_mask: [B, S] with True = masked
        B, S, D = tokens.shape
        q = self.q.expand(B, 1, D)                                    # [B,1,D]
        k = self.k_proj(tokens)                                       # [B,S,D]
        v = self.v_proj(tokens)                                       # [B,S,D]
        scores = (q @ k.transpose(1, 2)) / math.sqrt(D)               # [B,1,S]
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float('-inf'))
        w = torch.softmax(scores, dim=-1)                             # [B,1,S]
        pooled = (w @ v).squeeze(1)                                   # [B,D]
        return pooled

def chk(name, t):
    if torch.isnan(t).any() or torch.isinf(t).any():
        print(f"[NaN/Inf] in {name}")