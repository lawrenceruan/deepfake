"""Conservative FF++ paired-clip preparation; raw files are opened read-only.

Run with python -m deepfake.preprocess --help. Correspondence checks are
automatic quality gates, not human ground-truth certification.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import PreparedClipDataset, audit_splits
from .losses import dssim_map

CONVENTION = 'rgb_mean_half_dssim_gaussian11_sigma1.5'


def center_geometry(real_shape, fake_shape):
    rh, rw = real_shape[:2]
    fh, fw = fake_shape[:2]
    if rh != fh or rw < fw:
        raise ValueError('Unsupported geometry: require equal height and real width >= fake width')
    left = (rw - fw) // 2
    return [left, 0, left + fw, fh]


def crop(image, box):
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def resolve_input(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing input or path outside raw root: {name}')
    return path


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def validate_output(root, out):
    root, out = root.resolve(), out.resolve()
    if not root.is_dir() or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('Output must be disjoint from the existing raw-data root')
    if out.exists():
        raise ValueError('Output already exists; use a new run directory to avoid overwriting data')


def decode_indices(path, indices, required_indices=None):
    """Sequential decode: index correspondence is independent of container FPS."""
    wanted = set(indices)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {path}')
    metadata = dict(fps=cap.get(cv2.CAP_PROP_FPS),
                    header_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frames, timestamps = {}, {}
    try:
        for i in range(max(wanted) + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamps[i] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    finally:
        cap.release()
    missing = set(indices if required_indices is None else required_indices) - frames.keys()
    if missing:
        raise ValueError(f'Unavailable frame indices: {sorted(missing)}')
    return frames, timestamps, metadata


def face_tube(frames, margin=1.4):
    """Reference-only Haar detections; fixed square tube avoids crop jitter.

    Deliberately rejects missing/ambiguous faces instead of silently guessing.
    A stronger detector can replace this smoke-test implementation later.
    """
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if detector.empty():
        raise ValueError('OpenCV face cascade is unavailable')
    boxes = []
    h, w = frames[0].shape[:2]
    for image in frames:
        if image.shape[:2] != (h, w):
            raise ValueError('Frame geometry changed inside clip')
        scale = min(1., 640 / max(h, w))
        small = cv2.resize(image, None, fx=scale, fy=scale)
        found = detector.detectMultiScale(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY),
                                         scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        found = sorted(found, key=lambda b: int(b[2]) * int(b[3]), reverse=True)
        if not found or (len(found) > 1 and found[1][2]*found[1][3] > .5*found[0][2]*found[0][3]):
            raise ValueError('Missing face or ambiguous multiple faces in reference')
        x, y, bw, bh = (np.asarray(found[0], dtype=float) / scale).tolist()
        boxes.append([x, y, x+bw, y+bh])
    a = np.array(boxes)
    centers = (a[:, :2] + a[:, 2:]) / 2
    widths = a[:, 2:] - a[:, :2]
    if len(a) > 1 and np.max(np.linalg.norm(np.diff(centers, axis=0), axis=1)) > .5*np.median(widths):
        raise ValueError('Face track jumped between sampled frames')
    lo, hi = a[:, :2].min(0), a[:, 2:].max(0)
    side = int(np.ceil(max(hi-lo) * margin))
    if side > min(h, w):
        raise ValueError('Face tube does not fit frame without padding')
    origin = np.round((lo+hi-side)/2).astype(int)
    origin = np.clip(origin, [0, 0], [w-side, h-side])
    tube = [int(origin[0]), int(origin[1]), int(origin[0]+side), int(origin[1]+side)]
    return tube, boxes


def check_correspondence(real, fake, indices, tube, max_mae=.04, max_p95=.12):
    """Check background outside the face tube, plus local time-offset alternatives.

    Thresholds are in RGB [0,1]. Reject low-texture background and clearly
    better neighboring frames. Near ties are reported (static scenes cannot
    prove exact timing). No optical flow or deformable image alignment.
    """
    h, w = fake[indices[0]].shape[:2]
    mask_full = np.ones((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = tube
    # Extra margin keeps manipulated boundaries out of the alignment score.
    pad = max(4, int(.1 * (x1-x0)))
    mask_full[max(0,y0-pad):min(h,y1+pad), max(0,x0-pad):min(w,x1+pad)] = 0
    mask = cv2.resize(mask_full, (128, 96), interpolation=cv2.INTER_NEAREST).astype(bool)
    if mask.mean() < .15:
        raise ValueError('Insufficient background support for correspondence check')
    def thumb(frame):
        return cv2.resize(frame, (128,96), interpolation=cv2.INTER_AREA).astype(np.float32)/255
    rsmall = {i: thumb(im) for i, im in real.items()}
    details = []
    for i in indices:
        f = thumb(fake[i])
        error = np.abs(rsmall[i][mask] - f[mask])
        mae, p95 = float(error.mean()), float(np.quantile(error, .95))
        texture = float(rsmall[i][mask].std(axis=0).mean())
        alternatives = {str(j): float(np.abs(rsmall[j][mask]-f[mask]).mean())
                        for j in range(max(0,i-2), i+3) if j in rsmall and j != i}
        best_other = min(alternatives.values(), default=mae)
        timing_conflict = best_other + .002 < mae and best_other < .8 * mae
        passed = mae <= max_mae and p95 <= max_p95 and texture >= .02 and not timing_conflict
        details.append(dict(frame_index=i, background_mae=mae, background_p95=p95,
                            background_texture_std=texture, neighboring_frame_mae=alternatives,
                            timing_ambiguous=abs(best_other-mae) <= .002, passed=passed))
    return dict(passed=all(r['passed'] for r in details),
                method='same_index_center_crop_background_and_local_offsets_v1',
                max_mae=max_mae, max_p95=max_p95, background_fraction=float(mask.mean()),
                caveat='Automatic screening; static background can leave timing ambiguous.',
                frames=details)


def to_tensor(frames, tube, size):
    images = [cv2.resize(crop(im, tube), (size,size), interpolation=cv2.INTER_AREA) for im in frames]
    return torch.from_numpy(np.stack(images)).permute(0,3,1,2).float().div_(255)


def save_preview(path, real, fake, maps):
    """Three time points, columns: reference, fake, fixed-scale DSSIM heatmap."""
    rows = []
    for i in sorted(set([0, len(real)//2, len(real)-1])):
        r, f = [(x[i].permute(1,2,0).numpy()*255).round().astype(np.uint8) for x in (real,fake)]
        heat = cv2.applyColorMap((maps[i,0].numpy()*255).round().astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        panels = []
        for label, panel in [('Real',r), ('Fake',f), ('DSSIM [0,1]',heat)]:
            header = np.zeros((28, panel.shape[1], 3), np.uint8)
            cv2.putText(header, f'{label} t={i}', (5,19), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
            panels.append(np.concatenate([header,panel]))
        rows.append(np.concatenate(panels, axis=1))
    if not cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows), cv2.COLOR_RGB2BGR)):
        raise IOError(f'Could not write {path}')


def prepare_pair(root, out, row, args, ordinal):
    real_path = resolve_input(root, row['reference_path'])
    fake_path = resolve_input(root, row['video_path'])
    source, donor = fake_path.stem.split('_')
    if real_path.stem != source or real_path.parent.name != 'original':
        raise ValueError('Reference must be the FF++ target original (first filename ID)')
    indices = getattr(args, 'frame_indices', None)
    if indices is None:
        indices = [args.start_frame + k*args.frame_stride for k in range(args.frames)]
    if len(indices) != args.frames or not indices or min(indices) < 0 or indices != sorted(indices):
        raise ValueError('Expected ordered nonnegative frame indices matching frames')
    nearby = sorted(set(j for i in indices for j in range(max(0,i-2),i+3)))
    before = {str(p.relative_to(root)): fingerprint(p) for p in (real_path,fake_path)}
    # Future neighbors beyond EOF are optional; selected frames are mandatory.
    real, rt, rm = decode_indices(real_path, nearby, required_indices=indices)
    fake, ft, fm = decode_indices(fake_path, indices)
    geometry = center_geometry(real[indices[0]].shape, fake[indices[0]].shape)
    real_shape, fake_shape = real[indices[0]].shape, fake[indices[0]].shape
    if any(im.shape != real_shape for im in real.values()) or any(im.shape != fake_shape for im in fake.values()):
        raise ValueError('Video frame dimensions changed')
    real = {i: crop(im, geometry) for i, im in real.items()}
    tube, boxes = face_tube([real[i] for i in indices], args.face_margin)
    checks = check_correspondence(real, fake, indices, tube, args.max_background_mae, args.max_background_p95)
    report = dict(video_path=row['video_path'], reference_path=row['reference_path'],
                  frame_indices=indices, real_metadata=rm, fake_metadata=fm,
                  real_timestamps=[rt[i] for i in indices], fake_timestamps=[ft[i] for i in indices],
                  real_center_crop_xyxy=geometry, real_size=list(real_shape[:2][::-1]),
                  fake_size=list(fake_shape[:2][::-1]), face_tube_xyxy=tube,
                  reference_face_boxes=boxes, alignment=checks, input_fingerprints=before)
    after = {str(p.relative_to(root)): fingerprint(p) for p in (real_path,fake_path)}
    if before != after:
        raise RuntimeError('Raw inputs changed during preprocessing')
    report['raw_inputs_unchanged'] = True
    if not checks['passed']:
        report['status'] = 'rejected_alignment'
        return [], report
    reference = to_tensor([real[i] for i in indices], tube, args.size)
    video = to_tensor([fake[i] for i in indices], tube, args.size)
    maps = dssim_map(reference, video)
    validity = torch.ones_like(maps, dtype=torch.bool)
    # Exclude 5-pixel SSIM window support at synthetic crop boundaries.
    validity[..., :5, :] = False
    validity[..., -5:, :] = False
    validity[..., :, :5] = False
    validity[..., :, -5:] = False
    frame_validity = torch.ones(args.frames, dtype=torch.bool)
    stem = f'{ordinal:05d}_{fake_path.parent.name}_{fake_path.stem}_s{args.start_frame}'
    base = dict(split=args.split, source_video_id=source, pair_id=stem,
                donor_sequence_id=donor, identity_ids=row.get('identity_ids', []),
                generator_family=fake_path.parent.name, alignment_verified=True,
                alignment_verification=checks['method'], map_convention=CONVENTION,
                raw_video_path=row['video_path'], raw_reference_path=row['reference_path'],
                frame_indices=indices, geometry_report=f'reports/{stem}.json')
    rows = []
    for kind, image, target, label in [('paired_fake',video,maps,1), ('real',reference,torch.zeros_like(maps),0)]:
        name = f'{stem}_{kind}'
        payload = dict(video=image, map_target=target, map_validity=validity, frame_validity=frame_validity)
        torch.save(payload, out/'clips'/f'{name}.pt')
        sample = dict(base, sample_id=name, tensor_path=f'clips/{name}.pt', sample_type=kind, visual_label=label)
        sample['raw_video_path'] = row['video_path'] if label else row['reference_path']
        rows.append(sample)
    save_preview(out/'previews'/f'{stem}.png', reference, video, maps)
    report.update(status='accepted', sample_ids=[r['sample_id'] for r in rows],
                  dssim_mean=float(maps[validity].mean()), dssim_max=float(maps[validity].max()),
                  preview=f'previews/{stem}.png', map_valid_fraction=float(validity.float().mean()))
    (out/'reports'/f'{stem}.json').write_text(json.dumps(report, indent=2)+'\n')
    return rows, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True, help='Candidate-pair JSONL; explicit selection, no implicit full run')
    p.add_argument('--output', type=Path, required=True, help='New directory disjoint from raw root')
    p.add_argument('--split', default='smoke', help='smoke is for validation only; no automatic dataset split')
    p.add_argument('--frames', type=int, default=17)
    p.add_argument('--start-frame', type=int, default=32)
    p.add_argument('--frame-stride', type=int, default=2)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--face-margin', type=float, default=1.4)
    p.add_argument('--max-background-mae', type=float, default=.04)
    p.add_argument('--max-background-p95', type=float, default=.12)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.frames < 1 or args.start_frame < 0 or args.frame_stride < 1 or args.size < 32 or args.size % 32:
        p.error('Require positive frames/stride, nonnegative start, and size divisible by 32')
    if args.face_margin < 1 or args.threads < 1 or not 0 < args.max_background_mae <= 1 or not 0 < args.max_background_p95 <= 1:
        p.error('Invalid margin, threads, or correspondence thresholds')
    root, out = args.raw_root.resolve(), args.output.resolve()
    try:
        validate_output(root, out)
    except ValueError as exc:
        p.error(str(exc))
    candidates = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    if not candidates:
        p.error('Empty pair selection')
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    if not hasattr(cv2, 'CascadeClassifier'):
        p.error('OpenCV build lacks CascadeClassifier; install tests/requirements-preprocess.txt')
    out.mkdir(parents=True)
    for name in ('clips','reports','previews'):
        (out/name).mkdir()
    config = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (out/'config.json').write_text(json.dumps(dict(config, torch_version=str(torch.__version__),
        opencv_version=cv2.__version__, map_convention=CONVENTION),indent=2)+'\n')
    (out/'input_pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in candidates))
    reports, rows = [], []
    for i, row in enumerate(candidates):
        try:
            new_rows, report = prepare_pair(root, out, row, args, i)
            rows.extend(new_rows)
        except (ValueError, OSError, cv2.error) as exc:
            report = dict(video_path=row.get('video_path'), status='rejected', error=str(exc))
        reports.append(report)
        (out/'pair_reports.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(f"{i+1}/{len(candidates)} {row.get('video_path')}: {report['status']}", flush=True)
    manifest = out/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    audit_splits(manifest)
    if rows:
        dataset = PreparedClipDataset(manifest,args.split)
        for i in range(len(dataset)):
            dataset[i]
    summary = dict(candidate_pairs=len(candidates), accepted_pairs=len(rows)//2,
                   rejected_pairs=len(candidates)-len(rows)//2, prepared_clips=len(rows),
                   loader_verified=bool(rows), split=args.split,
                   note='Automatic alignment screening, not human certification. No official splits assigned.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if not rows:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
