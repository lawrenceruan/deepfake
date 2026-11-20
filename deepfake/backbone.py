"""Frozen official Wan2.2-TI2V-5B adapter. Loading real weights requires CUDA."""
from pathlib import Path
from contextlib import nullcontext
import torch
from torch import nn
from .config import DetectorConfig


class FrozenWanFeatures(nn.Module):
    def __init__(self, vae, backbone, text_context, config=DetectorConfig()):
        super().__init__()
        self.vae = vae  # Upstream VAE is a wrapper, not an nn.Module.
        self.vae_model = vae.model  # Register weights for parameter/device inspection.
        self.backbone = backbone
        self.config = config
        if text_context.ndim != 2 or not 0 < len(text_context) <= backbone.text_len:
            raise ValueError('Text context must be [tokens,text_dim] within text_len')
        if text_context.shape[1] != backbone.text_dim:
            raise ValueError('Wrong text embedding dimension')
        if max(config.feature_layers) >= len(backbone.blocks):
            raise ValueError('Requested feature layer outside backbone')
        if backbone.dim != config.backbone_dim:
            raise ValueError('backbone_dim disagrees with loaded checkpoint')
        self.register_buffer('text_context', text_context.detach(), persistent=False)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @classmethod
    def from_pretrained(cls, checkpoint_dir, context_path, config=DetectorConfig(), device='cuda'):
        if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
            raise RuntimeError('Official Wan extraction requires a CUDA GPU and FlashAttention')
        # Lazy imports: pure detector heads and their CPU tests do not import Wan.
        from wan.modules.model import WanModel
        from wan.modules.vae2_2 import Wan2_2_VAE
        root = Path(checkpoint_dir)
        vae = Wan2_2_VAE(vae_pth=str(root/'Wan2.2_VAE.pth'),
                         device=torch.device(device), dtype=torch.bfloat16)
        backbone = WanModel.from_pretrained(str(root), torch_dtype=torch.bfloat16).to(device)
        # Wan explicitly evaluates timestep embeddings in FP32 with autocast
        # disabled; keep these weights FP32 even when the main DiT is BF16.
        backbone.time_embedding.float()
        backbone.time_projection.float()
        if backbone.model_type != 'ti2v' or tuple(backbone.patch_size) != (1, 2, 2):
            raise ValueError('This adapter expects Wan2.2 TI2V with patch_size (1,2,2)')
        context = torch.load(context_path, map_location=device, weights_only=True)
        if isinstance(context, dict):
            context = context['context']
        return cls(vae, backbone, context, config)

    @torch.no_grad()
    def forward(self, video, frame_validity):
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError('Expected [B,T,3,H,W] RGB in [0,1]')
        b, frames, _, height, width = video.shape
        if height % 32 or width % 32:
            raise ValueError('Height and width must be divisible by 32')
        if not torch.isfinite(video).all() or video.min() < 0 or video.max() > 1:
            raise ValueError('Video must contain finite RGB in [0,1]')
        valid = frame_validity.bool()
        if valid.shape != (b, frames) or not valid.any(1).all():
            raise ValueError('Every sample needs at least one valid frame')
        if (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError('Only trailing padding is supported')
        device = next(self.backbone.parameters()).device
        if video.device != device:
            raise ValueError('Move videos to the extractor device before calling forward')
        per_sample = []
        lengths = []
        # Individual encoding/probing prevents padded frames from changing the
        # frozen backbone's global attention for shorter clips.
        for i in range(b):
            length = int(valid[i].sum())
            clip = video[i, :length]
            canonical = 1 + 4*((length-1+3)//4)
            if canonical > length:
                clip = torch.cat([clip, clip[-1:].expand(canonical-length, -1, -1, -1)])
            latent = self.vae.encode([(clip.permute(1, 0, 2, 3)*2-1).float()])[0]
            pt, ph, pw = self.backbone.patch_size
            if any(n % p for n, p in zip(latent.shape[1:], (pt, ph, pw))):
                raise ValueError('Latent grid is not divisible by backbone patch size')
            seq_len = (latent.shape[1]//pt)*(latent.shape[2]//ph)*(latent.shape[3]//pw)
            sigma = self.config.probe_sigma
            if sigma:
                # Same fixed probe noise for every same-shaped clip, independent
                # of ordering/batch position. sigma is POST-shift flow sigma.
                rng = torch.Generator(device=device).manual_seed(self.config.probe_seed)
                noise = torch.randn(latent.shape, generator=rng, device=device, dtype=latent.dtype)
                latent = (1-sigma)*latent + sigma*noise
            # Upstream flow scheduler convention: t = sigma * training timesteps.
            timestep = torch.tensor([sigma*self.config.num_train_timesteps], device=device)
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
            with amp:
                result = self.backbone([latent], t=timestep,
                    context=[self.text_context], seq_len=seq_len,
                    feature_layers=self.config.feature_layers)
            grid = tuple(int(v) for v in result['grid_sizes'][0])
            tokens = int(result['seq_lens'][0])
            if tokens != grid[0]*grid[1]*grid[2]:
                raise RuntimeError('Wan grid metadata is inconsistent')
            features = [v[0, :tokens].reshape(*grid, -1).permute(3, 0, 1, 2).detach()
                        for v in result['hidden_states']]
            per_sample.append(features)
            lengths.append(grid[0])
        max_time = max(lengths)
        grids = []
        for depth in range(len(self.config.feature_layers)):
            padded = []
            for fs in per_sample:
                g = fs[depth]
                pad = g.new_zeros(g.shape[0], max_time-g.shape[1], *g.shape[2:])
                padded.append(torch.cat([g, pad], dim=1))
            grids.append(torch.stack(padded))
        latent_validity = torch.arange(max_time, device=device)[None] < torch.tensor(lengths, device=device)[:, None]
        return grids, latent_validity
