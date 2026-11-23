"""Frozen VideoMAE-B: unpooled final encoder tokens, no reconstruction decoder.

Preserve the prepared crop and every valid frame. Odd clips repeat their last
frame to complete a two-frame tubelet. Interpolate the pretrained positional
grid in T/H/W for variable clip lengths and the configured image resolution.
"""
from pathlib import Path
import inspect
import json
import torch
from torch import nn
from torch.nn import functional as F
from .feature_cache import sha256_file


class VideoMAERGBEncoder(nn.Module):
    channels = 768

    def __init__(self, encoder, image_size=256, mean=(.485,.456,.406), std=(.229,.224,.225)):
        super().__init__()
        self.encoder = encoder
        self.image_size = image_size
        self.contract = None
        cfg = encoder.config
        if cfg.patch_size != 16 or cfg.tubelet_size != 2 or cfg.hidden_size != 768:
            raise ValueError('Expected VideoMAE-B with 16px patches, two-frame tubelets and 768 channels')
        if image_size < 16 or image_size % 16:
            raise ValueError('VideoMAE image size must be divisible by 16')
        self.native_grid = (cfg.num_frames//2, cfg.image_size//16, cfg.image_size//16)
        self.register_buffer('mean', torch.tensor(mean)[None,None,:,None,None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None,None,:,None,None], persistent=False)
        self.encoder.requires_grad_(False)
        self.train(False)

    @classmethod
    def from_pretrained(cls, checkpoint, config, device='cuda'):
        path = Path(checkpoint).resolve()
        required = ('model.safetensors', 'config.json', 'preprocessor_config.json')
        if not all((path/name).is_file() for name in required):
            raise FileNotFoundError(f'Incomplete local VideoMAE checkpoint: {path}')
        try:
            import transformers
            from transformers import VideoMAEModel
        except ImportError as exc:
            raise ImportError('Install requirements-videomae.txt for the VideoMAE branch') from exc
        prep = json.loads((path/'preprocessor_config.json').read_text())
        with torch.random.fork_rng(devices=[]):
            encoder, info = VideoMAEModel.from_pretrained(
                path, local_files_only=True, use_safetensors=True,
                attn_implementation='sdpa', output_loading_info=True)
        if info['missing_keys'] or info['mismatched_keys'] or info.get('error_msgs'):
            raise ValueError(f'Incomplete pretrained VideoMAE encoder: {info}')
        obj = cls(encoder, config.videomae_image_size, prep['image_mean'], prep['image_std'])
        obj.contract = dict(
            checkpoint_sha256={name:sha256_file(path/name) for name in required},
            transformers_version=transformers.__version__,
            encoder_source_sha256=sha256_file(inspect.getfile(VideoMAEModel)),
            image_size=config.videomae_image_size, trainable=False,
            feature='final_encoder_after_layernorm_unpooled',
            position_policy='native_grid_trilinear_align_corners_false_v1',
            odd_frames='repeat_last_valid_frame', normalization=prep,
            temporal_alignment='tubelet_centers_last_duplicate_center_at_last_frame_v1')
        return obj.to(device)

    def train(self, mode=True):
        super().train(False)
        return self

    def position_grid(self, grid, dtype, device):
        position = self.encoder.embeddings.position_embeddings.to(device=device)
        if tuple(grid) != self.native_grid:
            position = position.reshape(1,*self.native_grid,self.channels).permute(0,4,1,2,3)
            position = F.interpolate(position.float(), size=grid, mode='trilinear', align_corners=False)
            position = position.flatten(2).transpose(1,2)
        return position.to(dtype=dtype)

    @torch.no_grad()
    def forward(self, video, frame_validity):
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError('Expected RGB video [B,T,3,H,W]')
        b,t = video.shape[:2]
        valid = frame_validity.to(device=video.device,dtype=torch.bool)
        if valid.shape != (b,t) or not valid.any(1).all() or (valid[:,1:] & ~valid[:,:-1]).any():
            raise ValueError('Expected nonempty right-padded frame validity')
        outputs = []
        for i in range(b):
            n = int(valid[i].sum())
            x = video[i:i+1,:n].float()
            if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                raise ValueError('VideoMAE input must be finite RGB in [0,1]')
            if x.shape[-2:] != (self.image_size,self.image_size):
                x = F.interpolate(x.flatten(0,1), (self.image_size,self.image_size),
                                  mode='bilinear',align_corners=False,antialias=True)[None]
            x = (x-self.mean)/self.std
            if n % 2:
                x = torch.cat([x,x[:,-1:]],dim=1)
            # Use the pretrained Conv3d directly to support sizes beyond the
            # checkpoint's fixed input check, without changing its weights.
            patch = self.encoder.embeddings.patch_embeddings.projection(x.permute(0,2,1,3,4))
            grid = patch.shape[2:]
            tokens = patch.flatten(2).transpose(1,2)
            tokens = tokens + self.position_grid(grid,tokens.dtype,tokens.device)
            encoded = self.encoder.encoder(tokens,return_dict=True).last_hidden_state
            if self.encoder.layernorm is not None:
                encoded = self.encoder.layernorm(encoded)
            feature = encoded.reshape(1,*grid,self.channels).permute(0,1,4,2,3)
            pad = feature.new_zeros(1,(t+1)//2-grid[0],self.channels,*grid[1:])
            outputs.append(torch.cat([feature,pad],dim=1))
        return [torch.cat(outputs)]
