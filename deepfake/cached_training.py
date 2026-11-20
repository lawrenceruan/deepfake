"""Read-only, bounded-memory access to an existing verified Wan cache."""
import json
from pathlib import Path
import torch
from .feature_cache import CachedWanFeatureDataset, load_features, sha256_file


class DiskWanFeatureCache:
    def __init__(self, root, manifest, context, config):
        self.root = Path(root).resolve()
        self.manifest = Path(manifest).resolve()
        metadata = json.loads((self.root/'metadata.json').read_text())
        contract = metadata['contract']
        expected = {k: config.to_dict()[k] for k in
                    ('feature_layers','backbone_dim','probe_sigma','probe_seed','num_train_timesteps')}
        expected['feature_layers'] = list(expected['feature_layers'])
        if contract['extraction'] != expected:
            raise ValueError('Wan cache extraction settings do not match')
        if contract['source_manifest_sha256'] != sha256_file(self.manifest):
            raise ValueError('Wan cache source manifest does not match')
        if contract['context_sha256'] != sha256_file(context):
            raise ValueError('Wan cache text context does not match')
        self.extraction_id = metadata['extraction_id']
        self.feature_layers = list(config.feature_layers)
        self.backbone_dim = config.backbone_dim
        self.records = {}
        for split in ('train','val'):
            ds = CachedWanFeatureDataset(self.root, split, self.extraction_id)
            self.records.update({(split,r['sample_id']):r for r in ds.rows})
        self.verified_sources = {}

    def get_batch(self, batch):
        if len(batch['video']) != 1:
            raise ValueError('Disk Wan cache currently requires batch size 1')
        row = batch['metadata'][0]
        record = self.records[(row['split'],row['sample_id'])]
        source = (self.manifest.parent/row['tensor_path']).resolve()
        cached_source = Path(record['tensor_path']).resolve()
        if source != cached_source:
            raise ValueError('Cached clip source path mismatch')
        stat = source.stat()
        stamp = (stat.st_size,stat.st_mtime_ns,stat.st_ino)
        if source not in self.verified_sources:
            if sha256_file(source) != record['source_tensor_sha256']:
                raise ValueError('Prepared clip checksum differs from Wan cache')
            self.verified_sources[source] = stamp
        elif self.verified_sources[source] != stamp:
            raise ValueError('Prepared clip changed during this run')
        payload = load_features(self.root, record, self.extraction_id)
        if list(payload['feature_layers']) != self.feature_layers or any(
                g.shape[0] != self.backbone_dim for g in payload['grids']):
            raise ValueError('Cached Wan feature layers/channels mismatch')
        size = [batch['video'].shape[1],*batch['video'].shape[-2:]]
        if list(payload['output_size']) != size or not torch.equal(
                payload['frame_validity'],batch['frame_validity'][0].cpu()):
            raise ValueError('Cached Wan geometry/frame mask differs from RGB clip')
        device = batch['video'].device
        return ([g[None].to(device) for g in payload['grids']],
                payload['latent_validity'][None].to(device))
