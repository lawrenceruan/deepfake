"""Trainable modules. These depend only on PyTorch, not Wan/CUDA packages."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class DepthFusion(nn.Module):
    def __init__(self, layers, in_dim, dim):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim))
            for _ in layers])
        self.fuse = nn.Linear(dim * len(layers), dim)

    def forward(self, grids):
        # B,C,T,H,W -> B,T,H,W,D
        if len(grids) != len(self.projections):
            raise ValueError('Incorrect number of feature depths')
        if any(g.shape != grids[0].shape for g in grids):
            raise ValueError('Feature depths must share a grid')
        return self.fuse(torch.cat([
            p(g.permute(0, 2, 3, 4, 1).float())
            for p, g in zip(self.projections, grids)], dim=-1))


def position_encoding(length, dim, device, dtype):
    # Fixed sinusoids on the actual grid; no maximum sequence length.
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, device=device).float()
                     * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq[:dim // 2])
    return pe.to(dtype)


class SpatialTemporalBlock(nn.Module):
    def __init__(self, dim, heads, ratio=4, dropout=0.0):
        super().__init__()
        self.norm_s, self.norm_t, self.norm_f = [nn.LayerNorm(dim) for _ in range(3)]
        self.spatial = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.temporal = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ratio), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * ratio, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid):
        b, t, h, w, d = x.shape
        mask = valid[:, :, None, None, None]
        x = x.masked_fill(~mask, 0)
        s = self.norm_s(x).reshape(b * t, h * w, d)
        y = self.spatial(s, s, s, need_weights=False)[0]
        x = (x + self.drop(y.reshape(b, t, h, w, d))).masked_fill(~mask, 0)
        q = self.norm_t(x).permute(0, 2, 3, 1, 4).reshape(b * h * w, t, d)
        padding = (~valid)[:, None, None, :].expand(b, h, w, t).reshape(b*h*w, t)
        y = self.temporal(q, q, q, key_padding_mask=padding, need_weights=False)[0]
        y = y.reshape(b, h, w, t, d).permute(0, 3, 1, 2, 4)
        x = (x + self.drop(y)).masked_fill(~mask, 0)
        return (x + self.drop(self.ffn(self.norm_f(x)))).masked_fill(~mask, 0)


class FrameRefine(nn.Module):
    """Spatial refinement per frame, avoiding full-resolution 3D activations."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(math.gcd(out_channels, 8), out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GELU())

    def forward(self, x, size):
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
        y = self.net(F.interpolate(y, size=size, mode='bilinear', align_corners=False))
        return y.reshape(b, t, y.shape[1], *size).permute(0, 2, 1, 3, 4)


class MapDecoder(nn.Module):
    def __init__(self, dim, channels=64):
        super().__init__()
        self.context = nn.Sequential(nn.Conv3d(dim, channels, 3, padding=1), nn.GELU())
        self.stages = nn.ModuleList([
            FrameRefine(channels, channels),
            FrameRefine(channels, channels // 2),
            FrameRefine(channels // 2, channels // 4)])
        self.output = nn.Conv3d(channels // 4, 1, 1)

    @staticmethod
    def restore_time(x, frames):
        # Wan2.2 causal temporal grid: sample 0, then endpoints 4,8,... .
        # Explicit interpolation at input frame indices (not generic resize T).
        pos = torch.arange(frames, device=x.device).float() / 4
        lo = pos.floor().long().clamp(max=x.shape[2]-1)
        hi = (lo + 1).clamp(max=x.shape[2]-1)
        alpha = (pos - lo).clamp(0, 1)[None, None, :, None, None].to(x.dtype)
        return x[:, :, lo] * (1-alpha) + x[:, :, hi] * alpha

    def forward(self, grid, output_size):
        t, h, w = output_size
        x = self.context(grid)
        # Spatial stages at /16, /8 and /4; restore time before final stage.
        for stage, divisor in zip(self.stages, (16, 8, 4)):
            if divisor == 4:
                x = self.restore_time(x, t)
            x = stage(x, (max(1, h//divisor), max(1, w//divisor)))
        logits = self.output(x)
        b = logits.shape[0]
        logits = F.interpolate(logits[:, 0].reshape(b*t, 1, *logits.shape[-2:]),
                               size=(h, w), mode='bilinear', align_corners=False)
        return logits.reshape(b, t, 1, h, w).sigmoid()


class RGBMapDecoder(MapDecoder):
    """Frame-aligned SAM skips at /16,/8,/4; Wan carries temporal context."""
    def __init__(self, dim, channels=64):
        super().__init__(dim, channels)
        self.rgb_adapters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, c, 1), nn.GroupNorm(math.gcd(c, 8), c), nn.GELU())
            for c in (channels, channels, channels//2)])

    def forward(self, grid, output_size, rgb_features):
        t, h, w = output_size
        if len(rgb_features) != 3:
            raise ValueError('Expected fine/middle/coarse RGB pyramid')
        # All RGB levels retain every valid frame. Unlike the Wan-only decoder,
        # restore time before the first spatial stage rather than the last.
        x = self.restore_time(self.context(grid), t)
        for stage, adapter, rgb, divisor in zip(self.stages, self.rgb_adapters,
                                                reversed(rgb_features), (16,8,4)):
            b, c, _, _, _ = x.shape
            if rgb.ndim != 5 or rgb.shape[:3] != (b,t,256):
                raise ValueError('RGB pyramid must have shape [B,T,256,H,W]')
            size = (max(1,h//divisor), max(1,w//divisor))
            rgb = F.interpolate(rgb.flatten(0,1), size=size, mode='bilinear', align_corners=False)
            skip = adapter(rgb).reshape(b,t,c,*size).permute(0,2,1,3,4)
            frames = x.permute(0,2,1,3,4).reshape(b*t,c,*x.shape[-2:])
            frames = F.interpolate(frames, size=size, mode='bilinear', align_corners=False)
            x = frames.reshape(b,t,c,*size).permute(0,2,1,3,4) + skip
            x = stage(x, size)
        logits = self.output(x)[:,0].reshape(b*t,1,*x.shape[-2:])
        logits = F.interpolate(logits, size=(h,w), mode='bilinear', align_corners=False)
        return logits.reshape(b,t,1,h,w).sigmoid()


class VideoMAEMapDecoder(MapDecoder):
    """Adapt one compact VideoMAE grid, then resize at each decoder stage."""
    def __init__(self, dim, channels=64):
        super().__init__(dim, channels)
        self.rgb_adapters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(768,c,1), nn.GroupNorm(math.gcd(c,8),c), nn.GELU())
            for c in (channels,channels,channels//2)])

    @staticmethod
    def restore_rgb_time(x, frames):
        if x.shape[2] != (frames+1)//2:
            raise ValueError('VideoMAE grid does not match valid frame length')
        # Centers of frame pairs: .5,2.5,... . A duplicated final frame has
        # its center at its actual frame index, not half a frame later.
        centers = torch.arange(x.shape[2],device=x.device,dtype=torch.float32)*2+.5
        if frames % 2:
            centers[-1] = frames-1
        positions = torch.arange(frames,device=x.device,dtype=torch.float32)
        hi = torch.searchsorted(centers,positions).clamp(max=len(centers)-1)
        lo = (hi-1).clamp(min=0)
        alpha = ((positions-centers[lo])/(centers[hi]-centers[lo]).clamp(min=1e-6)).clamp(0,1)
        alpha = alpha[None,None,:,None,None].to(x.dtype)
        return x[:,:,lo]*(1-alpha)+x[:,:,hi]*alpha

    def forward(self, grid, output_size, rgb_features):
        t,h,w = output_size
        if len(rgb_features) != 1:
            raise ValueError('Expected a single unpooled VideoMAE feature grid')
        rgb = rgb_features[0]
        b = grid.shape[0]
        if rgb.ndim != 5 or rgb.shape[:3] != (b,(t+1)//2,768):
            raise ValueError('Expected VideoMAE [B,ceil(T/2),768,H,W]')
        x = self.restore_time(self.context(grid),t)
        for stage,adapter,divisor in zip(self.stages,self.rgb_adapters,(16,8,4)):
            size = (max(1,h//divisor),max(1,w//divisor))
            skip = adapter(rgb.flatten(0,1))
            c = skip.shape[1]
            skip = skip.reshape(b,rgb.shape[1],c,*skip.shape[-2:]).permute(0,2,1,3,4)
            skip = self.restore_rgb_time(skip,t).permute(0,2,1,3,4).flatten(0,1)
            skip = F.interpolate(skip,size=size,mode='bilinear',align_corners=False)
            skip = skip.reshape(b,t,c,*size).permute(0,2,1,3,4)
            x = F.interpolate(x.permute(0,2,1,3,4).flatten(0,1),size=size,
                              mode='bilinear',align_corners=False)
            x = x.reshape(b,t,c,*size).permute(0,2,1,3,4)
            x = stage(x+skip,size)
        logits = self.output(x)[:,0].reshape(b*t,1,*x.shape[-2:])
        logits = F.interpolate(logits,size=(h,w),mode='bilinear',align_corners=False)
        return logits.reshape(b,t,1,h,w).sigmoid()
