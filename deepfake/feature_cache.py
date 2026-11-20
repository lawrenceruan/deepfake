"""Lossless on-disk Wan features, separate from prepared videos and targets."""
import hashlib
import io
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validate_payload(payload, extraction_id, sample_id):
    if payload['extraction_id'] != extraction_id or payload['sample_id'] != sample_id:
        raise ValueError('Feature identity/configuration mismatch')
    grids, valid = payload['grids'], payload['latent_validity']
    frame = payload['frame_validity']
    if valid.dtype != torch.bool or valid.ndim != 1 or not valid.any():
        raise ValueError('Invalid latent validity')
    if frame.dtype != torch.bool or frame.ndim != 1 or not frame.any():
        raise ValueError('Invalid frame validity')
    if (valid[1:] & ~valid[:-1]).any() or (frame[1:] & ~frame[:-1]).any():
        raise ValueError('Validity must be right padded')
    if len(grids) != len(payload['feature_layers']):
        raise ValueError('Incorrect feature layer count')
    for grid in grids:
        if grid.dtype != torch.float32 or grid.ndim != 4 or grid.shape[1] != len(valid):
            raise ValueError('Expected FP32 [channels, latent_time, height, width]')
        if not torch.isfinite(grid).all():
            raise ValueError('Non-finite features')


def save_features(root, row, payload, source_sha256):
    root = Path(root)
    validate_payload(payload, payload['extraction_id'], row['sample_id'])
    key = signature([row['split'], row['sample_id']])
    relative = Path('features') / key[:2] / (key + '.pt')
    path = root / relative
    sidecar = path.with_suffix('.json')
    if sidecar.exists():
        raise FileExistsError('Completed cache entry already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    restored = torch.load(temporary, map_location='cpu', weights_only=True)
    validate_payload(restored, payload['extraction_id'], row['sample_id'])
    for a, b in zip(payload['grids'], restored['grids']):
        if not torch.equal(a, b):
            raise RuntimeError('Feature serialization changed values')
    for name in ('latent_validity', 'frame_validity'):
        if not torch.equal(payload[name], restored[name]):
            raise RuntimeError('Feature serialization changed validity')
    digest = sha256_file(temporary)
    temporary.replace(path)
    record = dict(row, feature_path=str(relative), feature_sha256=digest,
                  source_tensor_sha256=source_sha256, extraction_id=payload['extraction_id'],
                  feature_shapes=[list(g.shape) for g in payload['grids']], feature_dtype='float32')
    write_json(sidecar, record)
    return record


def load_features(root, record, extraction_id):
    root = Path(root).resolve()
    path = (root / record['feature_path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Feature path outside cache')
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != record['feature_sha256']:
        raise ValueError('Feature checksum mismatch')
    payload = torch.load(io.BytesIO(blob), weights_only=True, map_location='cpu')
    validate_payload(payload, extraction_id, record['sample_id'])
    return payload


class CachedWanFeatureDataset(Dataset):
    """Unbatched grids plus validity and metadata; targets stay at tensor_path.

    For batch size one, add a leading dimension to every grid and both validity
    masks before passing them to DetectorHeads. RGB/DSSIM data remain external.
    """
    def __init__(self, root, split, expected_extraction_id=None):
        self.root = Path(root).resolve()
        summary = json.loads((self.root / 'complete.json').read_text())
        self.extraction_id = summary['extraction_id']
        if not summary['complete'] or (expected_extraction_id is not None and
                                      expected_extraction_id != self.extraction_id):
            raise ValueError('Incomplete or incompatible feature cache')
        manifest = self.root / 'manifest.jsonl'
        if sha256_file(manifest) != summary['manifest_sha256']:
            raise ValueError('Cache manifest checksum mismatch')
        rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
        if len(rows) != summary['clips'] or len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError('Cache manifest count or identity mismatch')
        self.rows = [r for r in rows if r['split'] == split]
        if not self.rows:
            raise ValueError('No cached samples for split')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        record = self.rows[index]
        return dict(load_features(self.root, record, self.extraction_id), metadata=record)
