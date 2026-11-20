"""Three-second FF++ segments sampled on reference timestamps, without MP4 cuts.

Uses the existing FF++ same-index correspondence checks and center-crop policy.
This entry point does not modify the older single-clip trial or its splits.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from .data import PreparedClipDataset, audit_splits
from .preprocessing_config import PreprocessingConfig
from .preprocess import (CONVENTION, fingerprint, prepare_pair, resolve_input,
                         validate_output)


def read_timestamps(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {path}')
    times = []
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.)
    finally:
        cap.release()
    times = np.asarray(times, dtype=float)
    if not len(times) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Need finite strictly increasing decoded timestamps; no silent FPS fallback')
    return times


def plan_segments(reference_times, duration=3., sample_fps=8.):
    """Inclusive sampling endpoints: adjacent full windows share one boundary.

    Never stretch short footage or sample beyond the last available timestamp.
    Low-FPS input may reuse a source frame at multiple real sampling times;
    such resampling is distinct from invalid trailing padding.
    """
    times = np.asarray(reference_times, dtype=float)
    if not len(times) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Invalid reference timeline')
    steps = duration * sample_fps
    if duration <= 0 or sample_fps <= 0 or not np.isfinite(steps) or not np.isclose(steps, round(steps)):
        raise ValueError('Duration times sampling FPS must be a positive integer')
    maximum = int(round(steps)) + 1
    if (maximum - 1) % 4:
        raise ValueError('Maximum frames must follow Wan 4n+1 convention')
    relative = times - times[0]
    plans = []
    for number in range(int(np.floor(relative[-1] / duration)) + 1):
        start = number * duration
        # Do not create a singleton tail containing only a shared endpoint.
        if number and start >= relative[-1] - 1e-8:
            break
        requested = start + np.arange(maximum) / sample_fps
        requested = requested[requested <= relative[-1] + 1e-8]
        right = np.searchsorted(relative, requested).clip(0, len(times)-1)
        left = (right-1).clip(0)
        selected = np.where(abs(relative[left]-requested) <= abs(relative[right]-requested), left, right)
        plans.append(dict(segment_index=number, segment_start_s=float(start),
                          segment_end_s=float(min(start+duration, relative[-1])),
                          segment_duration_s=float(duration), sampling_fps=float(sample_fps),
                          maximum_frames=maximum, valid_frames=len(selected),
                          frame_indices=selected.tolist(),
                          requested_timestamps_s=(requested+times[0]).tolist(),
                          selected_reference_timestamps_s=times[selected].tolist(),
                          repeated_source_samples=int(len(selected)-len(set(selected.tolist())))))
    return plans


def pad_payload(payload, maximum):
    valid = len(payload['video'])
    if not 0 < valid <= maximum:
        raise ValueError('Invalid segment frame count')
    result = dict(payload)
    result['video'] = torch.cat([payload['video'], payload['video'][-1:].expand(maximum-valid,-1,-1,-1)])
    for name in ('map_target','map_validity'):
        x = payload[name]
        result[name] = torch.cat([x, x.new_zeros((maximum-valid,*x.shape[1:]))])
    result['frame_validity'] = torch.arange(maximum) < valid
    result['map_validity'] &= result['frame_validity'][:,None,None,None]
    return result


def prepare_segment(root, out, row, args, ordinal, plan):
    local = SimpleNamespace(**vars(args))
    local.frame_indices = plan['frame_indices']
    local.frames = plan['valid_frames']
    local.start_frame = local.frame_indices[0]
    local.frame_stride = 1  # Explicit indices take precedence.
    local.split = row.get('split', args.split)
    samples, report = prepare_pair(root,out,row,local,ordinal)
    report['segment'] = plan
    for sample in samples:
        path = out / sample['tensor_path']
        payload = torch.load(path,weights_only=True,map_location='cpu')
        torch.save(pad_payload(payload,plan['maximum_frames']),path)
        sample.update(segment=plan, recording_pair_id=f"{Path(row['video_path']).parent.name}/{Path(row['video_path']).stem}",
                      preprocessing_version='ffpp_segments_v1', temporal_padding='repeat_last_invalid')
    if samples:
        (out/samples[0]['geometry_report']).write_text(json.dumps(report,indent=2)+'\n')
    return samples, report


def normalize_pairs(rows, default_split='smoke'):
    """Accept adapter pair rows or reprocess the recording selection of a manifest.

    Prepared tensors are never used as temporal source data. Real rows are
    skipped, and multiple existing clips collapse to one raw recording pair.
    """
    selected={}
    for row in rows:
        if 'tensor_path' in row:
            if row.get('sample_type')!='paired_fake':
                continue
            row=dict(video_path=row['raw_video_path'],reference_path=row['raw_reference_path'],
                     split=row['split'],identity_ids=row.get('identity_ids',[]))
        else:
            row=dict(row,split=row.get('split',default_split))
        key=(row['video_path'],row['reference_path'])
        if key in selected and selected[key]['split']!=row['split']:
            raise ValueError('Recording pair assigned to conflicting splits')
        if key in selected:
            row['identity_ids']=sorted(set(row.get('identity_ids',[])) |
                                       set(selected[key].get('identity_ids',[])))
        selected[key]=row
    return list(selected.values())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--pairs',type=Path,required=True,help='Adapter pair JSONL or existing prepared manifest (preserves recording selection/splits)')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/preprocess_3s.json')
    parser.add_argument('--duration',type=float,help='Optional override of configured segment duration')
    parser.add_argument('--sample-fps',type=float,help='Optional override of configured sampling FPS')
    parser.add_argument('--size',type=int,help='Optional override of configured crop size')
    parser.add_argument('--split',default='smoke')
    parser.add_argument('--face-margin',type=float,default=1.4)
    parser.add_argument('--max-background-mae',type=float,default=.04)
    parser.add_argument('--max-background-p95',type=float,default=.12)
    parser.add_argument('--max-segments-per-pair',type=int,default=0,help='0 processes all segments; positive values for smoke checks')
    a=parser.parse_args()
    cfg=PreprocessingConfig.load(a.config)
    overrides={key:value for key,value in [('segment_duration_s',a.duration),('sampling_fps',a.sample_fps),('size',a.size)] if value is not None}
    cfg=replace(cfg,**overrides)
    a.duration,a.sample_fps,a.size=cfg.segment_duration_s,cfg.sampling_fps,cfg.size
    if a.size<32 or a.size%32 or a.max_segments_per_pair<0 or a.face_margin<1:
        parser.error('Invalid size, segment limit, or face margin')
    if not 0<a.max_background_mae<=1 or not 0<a.max_background_p95<=1:
        parser.error('Invalid alignment thresholds')
    plan_segments([0.],a.duration,a.sample_fps)
    root,out=a.raw_root.resolve(),a.output.resolve()
    validate_output(root,out)
    pairs=normalize_pairs([json.loads(s) for s in a.pairs.read_text().splitlines() if s.strip()],a.split)
    if not pairs:
        parser.error('Empty pair selection')
    # Audit assigned splits before expensive processing, including rejected pairs.
    audit_rows=[]
    for r in pairs:
        source,donor=Path(r['video_path']).stem.split('_')
        audit_rows.append(dict(source_video_id=source,donor_sequence_id=donor,
                               identity_ids=r.get('identity_ids',[]),split=r.get('split',a.split)))
    torch.set_num_threads(4);cv2.setNumThreads(4)
    out.mkdir(parents=True)
    for name in ('clips','reports','previews'):
        (out/name).mkdir()
    (out/'input_split_audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in audit_rows))
    audit_splits(out/'input_split_audit.jsonl')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    (out/'config.json').write_text(json.dumps(dict(config,map_convention=CONVENTION,
        preprocessing_version='ffpp_segments_v1',endpoint_policy='adjacent full windows share one endpoint'),indent=2)+'\n')
    (out/'preprocessing_config.json').write_text(json.dumps(cfg.to_dict(),indent=2)+'\n')
    (out/'input_pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
    manifest,reports=[],[]
    for row in pairs:
        try:
            real=resolve_input(root,row['reference_path']);fake=resolve_input(root,row['video_path'])
            before={str(p):fingerprint(p) for p in (real,fake)}
            rt,ft=read_timestamps(real),read_timestamps(fake)
            # FF++ adapter: same-index candidates on the common prefix only.
            # Each selected segment must still pass the spatial/temporal gate.
            plans=plan_segments(rt[:min(len(rt),len(ft))],a.duration,a.sample_fps)
            if a.max_segments_per_pair:
                plans=plans[:a.max_segments_per_pair]
            for plan in plans:
                try:
                    samples,report=prepare_segment(root,out,row,a,len(reports),plan)
                    for sample in samples:
                        sample['preprocessing_config_id']=cfg.fingerprint
                        sample['preprocessing_config_path']='preprocessing_config.json'
                    manifest.extend(samples)
                except (ValueError,OSError,cv2.error) as exc:
                    report=dict(video_path=row['video_path'],status='rejected',error=str(exc),segment=plan)
                reports.append(report)
                (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
                (out/'segment_reports.json').write_text(json.dumps(reports,indent=2)+'\n')
                print(row['video_path'],plan['segment_index'],report['status'],flush=True)
            if before!={str(p):fingerprint(p) for p in (real,fake)}:
                raise RuntimeError('Raw inputs changed during segmentation')
        except (ValueError,OSError,cv2.error) as exc:
            reports.append(dict(video_path=row['video_path'],status='rejected_video',error=str(exc)))
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    (out/'segment_reports.json').write_text(json.dumps(reports,indent=2)+'\n')
    audit_splits(out/'manifest.jsonl')
    for split in {r['split'] for r in manifest}:
        ds=PreparedClipDataset(out/'manifest.jsonl',split)
        for i in range(len(ds)):
            ds[i]
    summary=dict(prepared_clips=len(manifest),accepted_segment_pairs=len(manifest)//2,
                 rejected_items=sum(r['status']!='accepted' for r in reports),
                 preprocessing_config_id=cfg.fingerprint,maximum_frames=cfg.maximum_frames,
                 loader_verified=bool(manifest),note='Automatic alignment screening, not exact timing certification')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if not manifest:
        raise SystemExit(2)


if __name__=='__main__':
    main()
