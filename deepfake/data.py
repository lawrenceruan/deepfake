"""Loader for pre-aligned, pre-cropped clip tensors. No silent face alignment."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .losses import dssim_map


class PreparedClipDataset(Dataset):
    """JSONL rows: sample_id, tensor_path, split, visual_label, sample_type.

    Tensor payload: video [T,3,H,W], optional reference, map_target,
    map_validity and frame_validity. Paired fakes require alignment_verified.
    RGB may be float [0,1], or uint8 with video_encoding='uint8_rgb_0_255';
    the latter is converted back to float [0,1] before model use.
    Paths are relative to the manifest, not the process working directory.
    """
    def __init__(self, manifest, split, expected_preprocessing_id=None):
        self.path = Path(manifest).resolve()
        self.rows = [r for line in self.path.read_text().splitlines() if line.strip()
                     for r in [json.loads(line)] if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No samples for split {split}')
        configurations = {r.get('preprocessing_config_id') for r in self.rows}
        if len(configurations)>1:
            raise ValueError('Mixed preprocessing configurations in one split')
        if expected_preprocessing_id is not None and configurations != {expected_preprocessing_id}:
            raise ValueError('Prepared data does not match requested preprocessing configuration; regenerate from raw videos')
        ids = [r['sample_id'] for r in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate sample_id')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        p = Path(row['tensor_path'])
        payload = torch.load(p if p.is_absolute() else self.path.parent/p,
                             weights_only=True, map_location='cpu')
        video = payload['video'].float()
        if payload.get('video_encoding') == 'uint8_rgb_0_255':
            if payload['video'].dtype != torch.uint8:
                raise ValueError('uint8 RGB encoding requires uint8 tensor')
            video = video / 255.
        elif payload.get('video_encoding') not in (None, 'float_rgb_0_1'):
            raise ValueError('Unknown video encoding')
        if video.ndim != 4 or video.shape[1] != 3 or video.shape[0] < 1:
            raise ValueError('video must be [T,3,H,W]')
        if not torch.isfinite(video).all() or video.min() < 0 or video.max() > 1:
            raise ValueError('Expected finite RGB in [0,1]')
        t, _, h, w = video.shape
        if h % 32 or w % 32:
            raise ValueError('Crop dimensions must be divisible by 32')
        frame = payload.get('frame_validity', torch.ones(t, dtype=torch.bool)).bool()
        if frame.shape != (t,) or not frame.any() or (frame[1:] & ~frame[:-1]).any():
            raise ValueError('Expected nonempty right-padded frame_validity')
        target = torch.zeros(t, 1, h, w)
        mask = payload.get('map_validity', torch.ones_like(target, dtype=torch.bool)).bool()
        label = int(row['visual_label'])
        kind = row['sample_type']
        if kind == 'real' and label == 0:
            pass
        elif kind == 'unpaired_fake' and label == 1:
            mask = torch.zeros_like(target, dtype=torch.bool)
        elif kind == 'paired_fake' and label == 1:
            if not row.get('alignment_verified', False):
                raise ValueError('Paired map supervision requires verified alignment')
            if 'map_target' in payload:
                if row.get('map_convention') != 'rgb_mean_half_dssim_gaussian11_sigma1.5':
                    raise ValueError('Declare the expected DSSIM convention for cached targets')
                target = payload['map_target'].float()
            elif 'reference' in payload:
                target = dssim_map(payload['reference'], video)
            else:
                raise ValueError('Paired fake needs reference or cached map_target')
        else:
            raise ValueError('sample_type/visual_label mismatch')
        if target.shape != (t,1,h,w) or mask.shape != target.shape:
            raise ValueError('Incorrect target/mask dimensions')
        mask = mask & frame[:, None, None, None]
        if not torch.isfinite(target[mask]).all() or (target[mask] < 0).any() or (target[mask] > 1).any():
            raise ValueError('Valid map targets must be in [0,1]')
        return dict(video=video, visual_label=torch.tensor(label, dtype=torch.float32),
                    map_target=target, map_validity=mask, frame_validity=frame, metadata=row)


def collate_clips(samples):
    spatial = {tuple(s['video'].shape[-2:]) for s in samples}
    if len(spatial) != 1:
        raise ValueError('Batch clips must use one spatial crop size')
    t = max(s['video'].shape[0] for s in samples)
    result = {}
    for key in ('video', 'map_target', 'map_validity', 'frame_validity'):
        padded = []
        for s in samples:
            x = s[key]
            padded.append(torch.cat([x, x.new_zeros((t-len(x), *x.shape[1:]))]))
        result[key] = torch.stack(padded)
    result['visual_label'] = torch.stack([s['visual_label'] for s in samples])
    result['metadata'] = [s['metadata'] for s in samples]
    return result


def audit_splits(manifest):
    """Reject shared original recordings/pairs/known identities across splits."""
    seen = {}
    for line in Path(manifest).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get('source_video_id'):
            raise ValueError('Every row requires source_video_id for leakage checks')
        keys = [('source', r['source_video_id'])]
        if r.get('recording_component') is not None:
            keys.append(('recording_component', r['recording_component']))
        if r.get('donor_sequence_id'):
            keys.append(('source', r['donor_sequence_id']))
        if r.get('pair_id'):
            keys.append(('pair', r['pair_id']))
        keys.extend(('identity', identity) for identity in r.get('identity_ids', []))
        for key in keys:
            if key in seen and seen[key] != r['split']:
                raise ValueError(f'Cross-split leakage: {key}')
            seen[key] = r['split']
