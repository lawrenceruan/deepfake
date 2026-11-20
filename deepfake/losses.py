import torch
from torch.nn import functional as F


def detector_loss(outputs, labels, targets, validity, frame_validity, map_weight=1.0):
    pred = outputs['modification_map'].float()
    if pred.shape != targets.shape or pred.shape != validity.shape:
        raise ValueError('Map prediction, target and validity shapes must match')
    mask = validity.bool() & frame_validity[:, :, None, None, None].bool()
    # torch.where excludes NaN placeholders outside supervised pixels too.
    error = torch.where(mask, pred-targets.float(), torch.zeros_like(pred)).abs()
    map_loss = error.sum() / mask.sum().clamp_min(1)
    cls = F.binary_cross_entropy_with_logits(outputs['visual_logit'].float(), labels.float())
    parts = {'loss': cls + map_weight*map_loss, 'classification': cls, 'map': map_loss}
    for name, label in [('real_map', 0), ('fake_map', 1)]:
        subset = mask & (labels == label)[:, None, None, None, None]
        parts[name] = torch.where(subset, error, 0).sum() / subset.sum().clamp_min(1)
    parts['valid_map_pixels'] = mask.sum()
    return parts


@torch.no_grad()
def dssim_map(real, fake, window=11, sigma=1.5):
    """RGB-channel-mean (1-SSIM)/2; fixed [0,1] scale, no threshold.

    Inputs [...,3,H,W] in [0,1]. Reflect-padded Gaussian window.
    This is the half-scaled convention, not DiffusionFF's 1-SSIM.
    Correspondence, crop alignment and invalid borders are caller concerns.
    """
    if real.shape != fake.shape or real.shape[-3] != 3:
        raise ValueError('Expected matching [...,3,H,W] tensors')
    if window % 2 != 1 or window < 3 or min(real.shape[-2:]) <= window//2:
        raise ValueError('Invalid SSIM window for this image')
    for x in (real, fake):
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError('DSSIM requires finite RGB values in [0,1]')
    shape = real.shape
    x, y = [v.float().reshape(-1, 3, *shape[-2:]) for v in (real, fake)]
    p = window//2
    axis = torch.arange(-p, p+1, device=x.device).float()
    kernel = torch.exp(-axis.square()/(2*sigma*sigma))
    kernel /= kernel.sum()
    weight = (kernel[:, None]*kernel[None, :])[None, None].expand(3, 1, -1, -1)
    def avg(v):
        return F.conv2d(F.pad(v, (p,p,p,p), mode='reflect'), weight, groups=3)
    mx, my = avg(x), avg(y)
    vx, vy = (avg(x*x)-mx*mx).clamp_min(0), (avg(y*y)-my*my).clamp_min(0)
    cov = avg(x*y)-mx*my
    ssim = ((2*mx*my+0.01**2)*(2*cov+0.03**2)
            / ((mx*mx+my*my+0.01**2)*(vx+vy+0.03**2)))
    return ((1-ssim)/2).mean(1, keepdim=True).clamp(0,1).reshape(*shape[:-3], 1, *shape[-2:])
