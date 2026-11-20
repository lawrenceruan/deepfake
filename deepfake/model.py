import torch
from torch import nn
from .config import DetectorConfig
from .modules import DepthFusion, SpatialTemporalBlock, MapDecoder, position_encoding


class DetectorHeads(nn.Module):
    """Shared task adaptation and two heads; accepts cached unpooled features."""
    def __init__(self, config=DetectorConfig()):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.fusion = DepthFusion(config.feature_layers, config.backbone_dim, d)
        self.position_projections = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(3)])
        self.blocks = nn.ModuleList([SpatialTemporalBlock(
            d, config.attention_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.attention_blocks)])
        self.classifier = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d),
                                        nn.GELU(), nn.Linear(d, 1))
        self.decoder = MapDecoder(d, config.decoder_channels)

    def forward(self, grids, latent_validity, output_size, frame_validity=None):
        x = self.fusion(grids)
        b, t, h, w, d = x.shape
        valid = latent_validity.to(device=x.device, dtype=torch.bool)
        if valid.shape != (b, t) or not valid.any(dim=1).all():
            raise ValueError('Each sample requires at least one valid latent frame')
        if (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError('Only right-padded latent sequences are supported')
        # Distinct projections avoid assigning identical positions to (h,w)
        # and (w,h), as a plain sum of the same sinusoids would do.
        tp, hp, wp = [p(position_encoding(n, d, x.device, x.dtype))
                      for p,n in zip(self.position_projections, (t,h,w))]
        pe = (tp[None, :, None, None] + hp[None, None, :, None]
              + wp[None, None, None, :])
        x = x + pe / 3**0.5
        for block in self.blocks:
            x = block(x, valid)
        pooled = x.mean(dim=(2, 3))
        pooled = (pooled * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        logit = self.classifier(pooled).squeeze(-1)
        if frame_validity is None:
            frame_validity = torch.ones(b, output_size[0], device=x.device, dtype=torch.bool)
        frame_validity = frame_validity.to(device=x.device, dtype=torch.bool)
        if frame_validity.shape != (b, output_size[0]) or not frame_validity.any(1).all():
            raise ValueError('Invalid frame_validity')
        if (frame_validity[:, 1:] & ~frame_validity[:, :-1]).any():
            raise ValueError('Only right-padded frames are supported')
        # Decode each unpadded sample independently: padding must not enter
        # temporal interpolation, convolutions or normalization statistics.
        maps = []
        for i in range(b):
            n = int(valid[i].sum())
            frames = int(frame_validity[i].sum())
            grid = x[i:i+1, :n].permute(0, 4, 1, 2, 3)
            pred = self.decoder(grid, (frames, *output_size[1:]))
            maps.append(torch.cat([pred, pred.new_zeros(
                1, output_size[0]-frames, 1, *output_size[1:])], dim=1))
        return {'visual_logit': logit, 'modification_map': torch.cat(maps)}


class WanOnlyDetector(nn.Module):
    def __init__(self, extractor, config=DetectorConfig()):
        super().__init__()
        self.extractor = extractor
        self.heads = DetectorHeads(config)
        self.config = config

    def train(self, mode=True):
        super().train(mode)
        self.extractor.eval()
        return self

    def forward(self, video, frame_validity=None):
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError('Expected video [B,T,3,H,W]')
        if frame_validity is None:
            frame_validity = torch.ones(video.shape[:2], dtype=torch.bool, device=video.device)
        with torch.no_grad():
            features, latent_validity = self.extractor(video, frame_validity)
        return self.heads(features, latent_validity,
                          (video.shape[1], *video.shape[-2:]), frame_validity)
