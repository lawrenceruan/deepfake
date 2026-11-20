"""Versioned, resumable full FF++ preparation and recording-component holdout."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
from types import SimpleNamespace

import cv2
import torch

from .data import PreparedClipDataset, audit_splits
from .preprocess import fingerprint, resolve_input, validate_output
from .preprocessing_config import PreprocessingConfig
from .segment_preprocess import read_timestamps, plan_segments, prepare_segment


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def compact_payload(payload, real=False):
    """Lossless for crops originating as uint8/255; targets stay float32."""
    result = dict(payload)
    video = (payload['video'] * 255).round().to(torch.uint8)
    if not torch.equal(video.float()/255, payload['video']):
        raise ValueError('Compact encoding would change RGB values')
    result.update(video=video, video_encoding='uint8_rgb_0_255')
    if real:
        del result['map_target']  # Loader reconstructs the exact all-zero target.
    return result


def recording_components(pairs):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for row in pairs:
        source, donor = Path(row['video_path']).stem.split('_')
        a, b = find(source), find(donor)
        parent[max(a,b)] = min(a,b)
    return {x: find(x) for x in sorted(parent)}


def split_rows(rows, pairs, eval_clips=50, seed=42):
    """One clip per held-out target/donor connected component; balanced labels.

    All other samples in the selected components are reserved, never trained.
    Selection uses metadata only, independently of detector outputs or DSSIM.
    """
    if eval_clips < 2 or eval_clips % 2:
        raise ValueError('Require a positive even evaluation count')
    components = recording_components(pairs)
    groups = defaultdict(list)
    for row in rows:
        groups[components[row['source_video_id']]].append(row)
    eligible = sorted(g for g, items in groups.items() if {r['visual_label'] for r in items} == {0,1})
    if len(eligible) < eval_clips:
        raise ValueError('Insufficient independent recording components')
    rng = random.Random(seed)
    selected = rng.sample(eligible, eval_clips)
    labels = [0]*(eval_clips//2) + [1]*(eval_clips//2)
    rng.shuffle(labels)
    chosen = set()
    for group, label in zip(selected, labels):
        candidates = [r for r in groups[group] if r['visual_label'] == label]
        full = [r for r in candidates if r['segment']['valid_frames'] == r['segment']['maximum_frames']]
        candidates = sorted(full or candidates, key=lambda r: r['sample_id'])
        chosen.add(rng.choice(candidates)['sample_id'])
    selected = set(selected)
    active, reserved = [], []
    for row in rows:
        row = dict(row, recording_component=components[row['source_video_id']])
        if row['sample_id'] in chosen:
            active.append(dict(row, split='val'))
        elif row['recording_component'] in selected:
            reserved.append(dict(row, split='reserved_holdout_relatives'))
        else:
            active.append(dict(row, split='train'))
    val = [r for r in active if r['split'] == 'val']
    assert len(val) == len({r['recording_component'] for r in val}) == eval_clips
    assert len({r['raw_video_path'] for r in val}) == eval_clips
    return active, reserved, dict(seed=seed, evaluation_clips=eval_clips,
        evaluation_labels=dict(Counter(str(r['visual_label']) for r in val)),
        selected_components=sorted(selected), component_count=len(set(components.values())),
        recording_to_component=components,
        selection_policy='one clip per component; balanced labels; prefer full windows; metadata only',
        note='Target and donor recording separation; not certified person-identity separation')


def split_rows_from_reference(rows, pairs, reference_rows, eval_clips=50, seed=42):
    """Reuse a previous method's holdout; never replace missing groups with training groups."""
    components = recording_components(pairs)
    reference = [r for r in reference_rows if r['split'] == 'val']
    if len(reference) != eval_clips or sum(r['visual_label'] for r in reference) != eval_clips//2:
        raise ValueError('Reference must contain the requested balanced evaluation clips')
    selected = {}
    for row in reference:
        source, donor = row['source_video_id'], row['donor_sequence_id']
        group = components.get(source)
        if group is None or components.get(donor) != group or row.get('recording_component', group) != group:
            raise ValueError('Reference and candidate recording components differ')
        if group in selected:
            raise ValueError('Reference requires one evaluation clip per component')
        selected[group] = row
    groups = defaultdict(list)
    for row in rows:
        groups[components[row['source_video_id']]].append(row)
    chosen, missing, matched = set(), [], []
    rng = random.Random(seed)
    for group, ref in sorted(selected.items()):
        candidates = [r for r in groups[group] if r['visual_label'] == ref['visual_label']]
        if not candidates:
            missing.append(dict(component=group, label=ref['visual_label'], reason='No accepted clip for reference label'))
            continue
        exact = [r for r in candidates if r['source_video_id'] == ref['source_video_id']
                 and r['donor_sequence_id'] == ref['donor_sequence_id']
                 and r['segment']['segment_index'] == ref['segment']['segment_index']]
        full = [r for r in candidates if r['segment']['valid_frames'] == r['segment']['maximum_frames']]
        choice = rng.choice(sorted(exact or full or candidates, key=lambda r: r['sample_id']))
        chosen.add(choice['sample_id'])
        matched.append(dict(component=group, sample_id=choice['sample_id'],
                            reference_sample_id=ref['sample_id'], same_parent_and_segment=bool(exact)))
    active, reserved = [], []
    for row in rows:
        row = dict(row, recording_component=components[row['source_video_id']])
        if row['sample_id'] in chosen:
            active.append(dict(row, split='val'))
        elif row['recording_component'] in selected:
            reserved.append(dict(row, split='reserved_holdout_relatives'))
        else:
            active.append(dict(row, split='train'))
    val = [r for r in active if r['split'] == 'val']
    assert len(val) == len({r['recording_component'] for r in val}) == len({r['raw_video_path'] for r in val})
    return active, reserved, dict(seed=seed, evaluation_clips=len(val), requested_evaluation_clips=eval_clips,
        evaluation_labels=dict(Counter(str(r['visual_label']) for r in val)),
        selected_components=sorted(selected), component_count=len(set(components.values())),
        recording_to_component=components, missing_evaluation_components=missing, matched_reference_clips=matched,
        selection_policy='fixed reference holdout and labels; prefer same parent/segment; never replace a missing group',
        note='Target and donor recording separation across methods; not certified person-identity separation')


def prepare_recording(task):
    root, out, row, cfg_dict = task
    root, out = Path(root), Path(out)
    cfg = PreprocessingConfig(**cfg_dict)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    pair_name = Path(row['video_path']).parent.name + '_' + Path(row['video_path']).stem
    folder = out/'pairs'/pair_name
    done = folder/'complete.json'
    if done.exists():
        return json.loads(done.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('clips','reports','previews'):
        (folder/name).mkdir(exist_ok=True)
    args = SimpleNamespace(split='unsplit', size=cfg.size, face_margin=1.4,
                           max_background_mae=.04, max_background_p95=.12)
    samples, reports = [], []
    real, fake = [resolve_input(root,row[key]) for key in ('reference_path','video_path')]
    before = {str(p.relative_to(root)):fingerprint(p) for p in (real,fake)}
    try:
        rt, ft = read_timestamps(real), read_timestamps(fake)
        plans = plan_segments(rt[:min(len(rt),len(ft))],cfg.segment_duration_s,cfg.sampling_fps)
        for plan in plans:
            try:
                new, report = prepare_segment(root,folder,row,args,plan['segment_index'],plan)
                for sample in new:
                    path = folder/sample['tensor_path']
                    payload = torch.load(path, weights_only=True, map_location='cpu')
                    torch.save(compact_payload(payload,sample['visual_label']==0),path)
                    for key in ('tensor_path','geometry_report'):
                        sample[key] = str(Path('pairs')/pair_name/sample[key])
                    sample['preprocessing_config_id'] = cfg.fingerprint
                    sample['preprocessing_config_path'] = 'preprocessing_config.json'
                    sample['storage_version'] = 'lossless_uint8_rgb_float32_dssim_v1'
                    samples.append(sample)
            except (ValueError, OSError, cv2.error) as exc:
                report = dict(status='rejected', error=str(exc),segment=plan)
            reports.append(report)
    except (ValueError, OSError, cv2.error) as exc:
        reports.append(dict(status='rejected_video',error=str(exc)))
    if before != {str(p.relative_to(root)):fingerprint(p) for p in (real,fake)}:
        raise RuntimeError('Raw input changed')
    write_rows(folder/'manifest.jsonl', [dict(s,tensor_path=str(out/s['tensor_path'])) for s in samples])
    if samples:
        ds = PreparedClipDataset(folder/'manifest.jsonl','unsplit',cfg.fingerprint)
        for i in range(len(ds)):
            ds[i]
    result = dict(video_path=row['video_path'], samples=samples, reports=reports,
                  input_fingerprints=before, raw_inputs_unchanged=True,loader_verified=bool(samples))
    write_json(done,result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--expected-raw-sha256',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/preprocess_3s.json'))
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--eval-clips',type=int,default=50)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--split-reference',type=Path,help='Reuse held-out recording groups and labels from this manifest')
    a = p.parse_args()
    cfg = PreprocessingConfig.load(a.config)
    root, out = a.raw_root.resolve(),a.output.resolve()
    pairs = [json.loads(s) for s in a.pairs.read_text().splitlines() if s.strip()]
    expected = json.loads(a.expected_raw_sha256.read_text())
    config = dict(raw_root=str(root),preprocessing=cfg.to_dict(),pairs=pairs,
                  expected_raw=expected,eval_clips=a.eval_clips,seed=a.seed,storage_version='lossless_uint8_rgb_float32_dssim_v1')
    reference_rows = None
    if a.split_reference:
        reference_rows = [json.loads(s) for s in a.split_reference.read_text().splitlines() if s.strip()]
        audit_splits(a.split_reference)
        # Validate group compatibility before expensive decoding, including absent groups.
        split_rows_from_reference([], pairs, reference_rows, a.eval_clips, a.seed)
        config['split_reference'] = dict(path=str(a.split_reference.resolve()), signature=fingerprint(a.split_reference))
    if a.resume:
        if json.loads((out/'run_config.json').read_text()) != config:
            raise ValueError('Resume configuration changed')
    else:
        validate_output(root,out)
        out.mkdir(parents=True)
        write_json(out/'run_config.json',config)
    for rel, signature in expected.items():
        if fingerprint(resolve_input(root,rel)) != signature:
            raise ValueError(f'Raw copy checksum mismatch: {rel}')
    write_json(out/'preprocessing_config.json',cfg.to_dict())
    write_rows(out/'input_pairs.jsonl',pairs)
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(prepare_recording,(str(root),str(out),row,cfg.to_dict())) for row in pairs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(dict(completed=len(results),total=len(pairs),video=result['video_path'],
                                  accepted_segments=len(result['samples'])//2)),flush=True)
    results.sort(key=lambda r:r['video_path'])
    rows = [s for r in results for s in r['samples']]
    active, reserved, split = (split_rows(rows,pairs,a.eval_clips,a.seed) if reference_rows is None else
                              split_rows_from_reference(rows,pairs,reference_rows,a.eval_clips,a.seed))
    write_rows(out/'all_prepared_manifest.jsonl',rows)
    write_rows(out/'manifest.jsonl',active)
    write_rows(out/'reserved_holdout_relatives.jsonl',reserved)
    audit_splits(out/'manifest.jsonl')
    for rel, signature in expected.items():
        if fingerprint(resolve_input(root,rel)) != signature:
            raise RuntimeError(f'Raw input modified: {rel}')
    summary = dict(complete=True,candidate_pairs=len(pairs),
        pairs_with_accepted_segments=sum(bool(r['samples']) for r in results),
        accepted_segment_pairs=len(rows)//2,prepared_clips=len(rows),
        active_clips=dict(Counter(r['split'] for r in active)),reserved_clips=len(reserved),
        active_valid_frames=dict((s,sum(r['segment']['valid_frames'] for r in active if r['split']==s)) for s in ('train','val')),
        segment_status_counts=dict(Counter(x['status'] for r in results for x in r['reports'])),
        rejection_reasons=dict(Counter(x.get('error',x['status']) for r in results for x in r['reports'] if x['status']!='accepted')),
        preprocessing_config_id=cfg.fingerprint,maximum_frames=cfg.maximum_frames,
        raw_sha256_verified_before_and_after=True,loader_verified=True,split_audit_passed=True,
        split=split)
    write_json(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
