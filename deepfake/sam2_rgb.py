"""Optional SAM 2.1 Hiera-Large image encoder and pretrained FPN, framewise.

Only the official image_encoder is instantiated: no mask/prompt decoder or
video memory. No import of SAM is needed for the Wan-only baseline or tests
that inject a small encoder. Checkpoint downloads are always explicit.
"""
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .feature_cache import sha256_file

SAM_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'


def load_image_encoder_state(encoder, checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or 'model' not in state:
        raise ValueError('Expected the official SAM 2.1 checkpoint with a model dictionary')
    prefix = 'image_encoder.'
    weights = {k[len(prefix):]: v for k, v in state['model'].items() if k.startswith(prefix)}
    if not weights:
        raise ValueError('Checkpoint contains no SAM image encoder weights')
    encoder.load_state_dict(weights, strict=True)


class SAM2RGBEncoder(nn.Module):
    channels = 256
    strides = (4, 8, 16)  # Official backbone_fpn order: fine -> coarse.

    def __init__(self, image_encoder, image_size=1024, frame_batch_size=1, trainable=False):
        super().__init__()
        self.image_encoder = image_encoder
        self.image_size = image_size
        self.frame_batch_size = frame_batch_size
        self.trainable = trainable
        self.contract = None
        if image_size < 64 or image_size % 32 or frame_batch_size < 1:
            raise ValueError('Invalid SAM image size/frame batch size')
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None], persistent=False)
        self.image_encoder.requires_grad_(trainable)
        self.train(False)

    @classmethod
    def from_pretrained(cls, checkpoint, config, device='cuda'):
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'SAM 2.1 Hiera-Large checkpoint not found: {path}')
        try:
            import sam2
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise ImportError('Install requirements-sam2.txt in a separate environment for the RGB branch') from exc
        # Instantiate just the encoder using the official configuration. Preserve
        # detector initialization RNG; frozen model construction must not alter it.
        config_path = Path(sam2.__file__).resolve().parent / SAM_CONFIG
        with torch.random.fork_rng(devices=[]):
            official = OmegaConf.load(config_path)
            encoder = instantiate(official.model.image_encoder)
            load_image_encoder_state(encoder, path)
        obj = cls(encoder, config.sam2_image_size, config.sam2_frame_batch_size, config.sam2_trainable)
        obj.contract = dict(checkpoint_sha256=sha256_file(path), config=SAM_CONFIG,
                            config_sha256=sha256_file(config_path),
                            image_size=config.sam2_image_size, trainable=config.sam2_trainable)
        package = Path(sam2.__file__).resolve().parent
        obj.contract['encoder_source_sha256'] = {name: sha256_file(package/name) for name in (
            'modeling/backbones/image_encoder.py', 'modeling/backbones/hieradet.py',
            'modeling/backbones/utils.py', 'modeling/position_encoding.py', 'modeling/sam2_utils.py')}
        return obj.to(device)

    def train(self, mode=True):
        super().train(mode if self.trainable else False)
        return self

    def forward(self, video, frame_validity):
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError('Expected RGB video [B,T,3,H,W]')
        b, t, _, h, w = video.shape
        valid = frame_validity.to(device=video.device, dtype=torch.bool)
        if valid.shape != (b, t) or not valid.any(1).all() or (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError('Expected nonempty right-padded frame validity')
        indices = valid.flatten().nonzero(as_tuple=True)[0]
        frames = video.flatten(0, 1).index_select(0, indices)
        if not torch.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError('SAM input must be finite RGB in [0,1]')
        levels = [[], [], []]
        # Resample each native FPN output to the crop's scale immediately, so a
        # frozen encoder does not retain 25 full-resolution 1024px pyramids.
        context = nullcontext() if self.trainable else torch.no_grad()
        with context:
            for chunk in frames.split(self.frame_batch_size):
                x = F.interpolate(chunk.float(), (self.image_size, self.image_size),
                                  mode='bilinear', align_corners=False, antialias=True)
                result = self.image_encoder((x-self.mean)/self.std)['backbone_fpn']
                if len(result) != 3:
                    raise ValueError('Expected three SAM 2.1 image-encoder FPN levels')
                for level, stride, features in zip(levels, self.strides, result):
                    if features.ndim != 4 or features.shape[:2] != (len(chunk), self.channels):
                        raise ValueError('Expected SAM FPN features [frames,256,H,W]')
                    level.append(F.interpolate(features, (max(1,h//stride), max(1,w//stride)),
                                               mode='bilinear', align_corners=False))
            outputs = []
            for level in levels:
                x = torch.cat(level)
                padded = x.new_zeros(b*t, *x.shape[1:]).index_copy(0, indices, x)
                outputs.append(padded.reshape(b, t, *x.shape[1:]))
        return outputs
