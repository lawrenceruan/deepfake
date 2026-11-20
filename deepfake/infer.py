"""Run a single prepared RGB clip; no reference or map is required."""
import argparse
from pathlib import Path
import torch
from .config import DetectorConfig
from .backbone import FrozenWanFeatures
from .model import WanOnlyDetector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--context', required=True)
    p.add_argument('--detector', required=True)
    p.add_argument('--clip', required=True, help='torch file with video [T,3,H,W] in [0,1]')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    state = torch.load(a.detector, weights_only=True, map_location='cpu')
    cfg = DetectorConfig(**state['config'])
    model = WanOnlyDetector(FrozenWanFeatures.from_pretrained(a.checkpoint_dir, a.context, cfg), cfg)
    model.heads.load_state_dict(state['heads'])
    model.heads.cuda()
    model.eval()
    clip = torch.load(a.clip, weights_only=True, map_location='cpu')
    video = clip['video'].float()
    if clip.get('video_encoding') == 'uint8_rgb_0_255':
        if clip['video'].dtype != torch.uint8:
            raise ValueError('uint8 encoding requires uint8 video')
        video = video / 255.
    elif clip.get('video_encoding') not in (None, 'float_rgb_0_1'):
        raise ValueError('Unknown video encoding')
    video = video.unsqueeze(0).cuda()
    valid = clip.get('frame_validity', torch.ones(video.shape[1], dtype=torch.bool))[None].cuda()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(video, valid)
    outputs = {k: v.float().cpu() for k,v in outputs.items()}
    outputs['fake_score'] = outputs['visual_logit'].sigmoid()
    outputs['frame_validity'] = valid.cpu()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, a.output)
    print({'fake_score': outputs['fake_score'].tolist(), 'output': a.output})


if __name__ == '__main__':
    main()
