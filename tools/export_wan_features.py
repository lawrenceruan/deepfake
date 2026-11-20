"""Independently export frozen Wan features; never attach to a running trainer."""
import argparse
from collections import Counter
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import time
import torch
from deepfake.backbone import FrozenWanFeatures
from deepfake.config import DetectorConfig
from deepfake.data import PreparedClipDataset, audit_splits
from deepfake.preprocessing_config import PreprocessingConfig
from deepfake.feature_cache import (CachedWanFeatureDataset, sha256_file, signature,
                                   write_json, save_features, load_features)


def stat_signature(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns, s.st_ino]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--limit', type=int, help='Separate smoke export only; never marks full export complete')
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error('--limit must be positive')
    run = json.loads(a.run_config.read_text())
    cfg = DetectorConfig(**run['detector'])
    manifest = Path(run['manifest']).resolve()
    audit_splits(manifest)
    prep = PreprocessingConfig.load(run['preprocessing_config'])
    datasets = {s: PreparedClipDataset(manifest, s, prep.fingerprint) for s in ('train', 'val')}
    entries = [(split, i, row) for split, ds in datasets.items() for i, row in enumerate(ds.rows)]
    # Visit representative methods/labels early without changing dataset splits.
    representative, rest, seen = [], [], set()
    for entry in entries:
        key = (entry[0], entry[2]['generator_family'], entry[2]['visual_label'])
        (representative if key not in seen else rest).append(entry)
        seen.add(key)
    entries = representative + rest
    if a.limit:
        entries = entries[:a.limit]
    out = a.output.resolve()
    protected = [manifest.parent, a.run_config.resolve().parent, Path(run['checkpoint_dir']).resolve()]
    sources = []
    for _, _, row in entries:
        path = (manifest.parent / row['tensor_path']).resolve()
        if path.is_relative_to(out) or out.is_relative_to(path.parent):
            raise ValueError('Cache output must be disjoint from prepared inputs')
        sources.append(path)
    if any(out.is_relative_to(path) or path.is_relative_to(out) for path in protected):
        raise ValueError('Cache output must be disjoint from existing data and run directories')
    if out.exists() and not a.resume:
        raise FileExistsError('Use a new cache directory or explicitly resume its exact configuration')
    if not torch.cuda.is_available():
        raise RuntimeError('Export requires a separately allocated CUDA GPU')
    checkpoint = Path(run['checkpoint_dir']).resolve()
    weights = sorted(checkpoint.glob('*.safetensors')) + [checkpoint/'Wan2.2_VAE.pth']
    weights += sorted(checkpoint.glob('*.json'))
    if not any(x.suffix == '.safetensors' for x in weights):
        raise ValueError('No Wan backbone weights found')
    runtime = Path(inspect.getfile(FrozenWanFeatures)).resolve().parent.parent
    code = sorted((runtime/'deepfake').glob('*.py')) + sorted((runtime/'wan').rglob('*.py'))
    dependencies = weights + [Path(run['context']).resolve(), manifest,
                              Path(run['preprocessing_config']).resolve(), a.run_config.resolve(),
                              Path(__file__).resolve()] + code
    dependency_stats = {str(x): stat_signature(x) for x in dependencies}
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'diffusers', 'flash-attn')}
    contract = dict(schema_version=1, preprocessing_id=prep.fingerprint,
        extraction={k: run['detector'][k] for k in ('feature_layers', 'backbone_dim', 'probe_sigma',
                     'probe_seed', 'num_train_timesteps')},
        weights_sha256={str(x): sha256_file(x) for x in weights},
        context_sha256=sha256_file(run['context']), source_manifest_sha256=sha256_file(manifest),
        extractor_source_sha256={str(x.relative_to(runtime)): sha256_file(x) for x in code},
        exporter_sha256=sha256_file(__file__),
        run_config_sha256=sha256_file(a.run_config), versions=versions,
        gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), autocast='bfloat16', stored_dtype='float32',
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, limit=a.limit,
        source_manifest=str(manifest), clips=len(entries), splits=['train', 'val'])
    extraction_id = signature(contract)
    metadata = dict(extraction_id=extraction_id, contract=contract)
    if a.resume:
        if json.loads((out/'metadata.json').read_text()) != metadata:
            raise ValueError('Resume extraction contract changed')
        if (out/'complete.json').exists():
            raise ValueError('Cache already complete; no write is necessary')
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        # Upper bound uses full 25-frame clips. Reserve 100 GiB for other work.
        required = len(entries)*3*3072*7*8*8*4
        if shutil.disk_usage(out.parent).free < required + 100*2**30:
            raise RuntimeError('Insufficient free space for lossless FP32 cache plus reserve')
        out.mkdir()
        write_json(out/'metadata.json', metadata)
        shutil.copy2(manifest, out/'source_manifest.jsonl')
        shutil.copy2(run['preprocessing_config'], out/'preprocessing_config.json')
    # A new process and a separately scheduled GPU; no signals or changes to training.
    torch.manual_seed(run['seed'])
    extractor = FrozenWanFeatures.from_pretrained(checkpoint, run['context'], cfg)
    extractor.eval()
    records = []
    started = time.monotonic()
    repeated_lengths = set()
    for number, ((split, index, row), source) in enumerate(zip(entries, sources), 1):
        key = signature([split, row['sample_id']])
        sidecar = out/'features'/key[:2]/(key+'.json')
        before = stat_signature(source)
        source_hash = sha256_file(source)
        if sidecar.exists():
            record = json.loads(sidecar.read_text())
            if record['source_tensor_sha256'] != source_hash or record['extraction_id'] != extraction_id:
                raise ValueError('Resumed cache input/configuration mismatch')
            load_features(out, record, extraction_id)
        else:
            item = datasets[split][index]
            video = item['video'].unsqueeze(0).cuda()
            frame = item['frame_validity'].unsqueeze(0).cuda()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                grids, valid = extractor(video, frame)
            payload = dict(sample_id=row['sample_id'], extraction_id=extraction_id,
                feature_layers=list(cfg.feature_layers),
                grids=[g[0].detach().cpu().contiguous() for g in grids],
                latent_validity=valid[0].cpu(), frame_validity=item['frame_validity'].clone(),
                output_size=[item['video'].shape[0], *item['video'].shape[-2:]])
            # Independently repeat one full and one short clip to check deterministic probing.
            length_kind = 'full' if bool(frame.all()) else 'short'
            if length_kind not in repeated_lengths:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    again, again_valid = extractor(video, frame)
                if not torch.equal(valid, again_valid) or not all(
                    torch.allclose(x, y, rtol=1e-5, atol=1e-6) for x, y in zip(grids, again)):
                    raise RuntimeError('Repeated frozen extraction is inconsistent')
                repeated_lengths.add(length_kind)
            record = save_features(out, row, payload, source_hash)
        if before != stat_signature(source):
            raise RuntimeError('Prepared input changed during export')
        records.append(record)
        if number % 100 == 0 or number == len(entries):
            if shutil.disk_usage(out).free < 100*2**30:
                raise RuntimeError('Stopped export to preserve 100 GiB free storage for other work')
            progress = dict(completed=number, total=len(entries), elapsed_seconds=time.monotonic()-started,
                            sample_id=row['sample_id'], extraction_id=extraction_id)
            write_json(out/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    if dependency_stats != {str(x): stat_signature(x) for x in dependencies}:
        raise RuntimeError('Extraction dependency changed during export')
    final = out/'manifest.jsonl'
    temporary = out/'manifest.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(r)+'\n' for r in records))
    temporary.replace(final)
    summary = dict(complete=True, scope='smoke' if a.limit else 'all_active_train_and_validation',
        clips=len(records), splits=dict(Counter(r['split'] for r in records)),
        methods=dict(Counter(r['generator_family'] for r in records)),
        extraction_id=extraction_id, manifest_sha256=sha256_file(final),
        serialized_features_exactly_verified=True, inputs_unchanged=True,
        repeat_checks=sorted(repeated_lengths), stored_dtype='float32',
        excluded='Reserved held-out relatives are not part of this active-run cache.')
    write_json(out/'complete.json', summary)
    try:
        for split in summary['splits']:
            ds = CachedWanFeatureDataset(out, split, extraction_id)
            ds[0]
    except Exception:
        (out/'complete.json').rename(out/'failed_final_check.json')
        raise
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
