"""Read-only audit of a local FF++ mirror; never labels alignment as verified.

Analysis dependencies: av, numpy. Writes JSON/CSV reports to --output,
not training targets.
"""
import argparse
import collections
import concurrent.futures
import csv
import json
from pathlib import Path
import time

import av
import numpy as np

FAMILIES = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']


def probe(path, root):
    row = dict(path=str(path.relative_to(root)), family=path.parent.name,
               bytes=path.stat().st_size, error=None)
    try:
        with av.open(str(path)) as container:
            streams = list(container.streams.video)
            row['video_streams'] = len(streams)
            row['audio_streams'] = len(container.streams.audio)
            if not streams:
                raise ValueError('No video stream')
            stream = streams[0]
            stream.codec_context.thread_count = 1
            row.update(width=stream.width, height=stream.height,
                fps=float(stream.average_rate) if stream.average_rate else None,
                base_fps=float(stream.base_rate) if stream.base_rate else None,
                time_base=str(stream.time_base), frames=stream.frames,
                duration_s=float(stream.duration*stream.time_base) if stream.duration else None,
                container_duration_s=container.duration/av.time_base if container.duration else None,
                codec=stream.codec_context.name,
                pixel_format=stream.codec_context.format.name if stream.codec_context.format else None,
                rotation=stream.metadata.get('rotate'),
                sample_aspect_ratio=str(stream.sample_aspect_ratio))
            frame = next(container.decode(stream))
            row.update(first_frame_decodes=True, first_pts=frame.pts,
                       first_time_s=float(frame.time) if frame.time is not None else None)
    except Exception as e:
        row.update(first_frame_decodes=False, error=f'{type(e).__name__}: {e}')
    return row


def stats(values):
    a = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(a):
        return None
    return {k: float(v) for k,v in zip(['min','p10','median','p90','max'],np.percentile(a,[0,10,50,90,100]))}


def metadata_audit(root, out, workers):
    paths = sorted(root.rglob('*.mp4'))
    start = time.time()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(lambda p: probe(p,root), paths):
            rows.append(row)
            if len(rows)%1000 == 0:
                print(f'Probed {len(rows)}/{len(paths)} ({time.time()-start:.1f}s)',flush=True)
    (out/'video_properties.json').write_text(json.dumps(rows,indent=2)+'\n')
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (out/'video_properties.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    summary={}
    for family in sorted(set(r['family'] for r in rows)):
        rs=[r for r in rows if r['family']==family]
        good=[r for r in rs if not r['error']]
        summary[family]=dict(videos=len(rs),size_gib=sum(r['bytes'] for r in rs)/2**30,
            total_hours=sum(r['duration_s'] or 0 for r in good)/3600,
            total_header_frames=sum(r['frames'] for r in good),
            frames=stats(r['frames'] for r in good),duration_s=stats(r['duration_s'] for r in good),
            fps=dict(collections.Counter(str(r['fps']) for r in good)),
            resolutions=dict(collections.Counter(f"{r['width']}x{r['height']}" for r in good)),
            codecs=dict(collections.Counter(r['codec'] for r in good)),
            pixel_formats=dict(collections.Counter(r['pixel_format'] for r in good)),
            audio_videos=sum(r['audio_streams']>0 for r in good),
            errors=[r for r in rs if r['error']])
    # Compare bundled CSV against actual headers, not just another CSV.
    indexed={r['path']:r for r in rows}
    csv_mismatches=[]
    for p in sorted((root/'csv').glob('*.csv')):
        if p.name.startswith('FF++') or p.name=='Mean_Data.csv':
            continue
        for r in csv.DictReader(p.open()):
            actual=indexed.get(r['File Path'])
            if actual and not actual['error']:
                for source,key in [('Frame Count','frames'),('Width','width'),('Height','height')]:
                    if int(r[source])!=actual[key]:
                        csv_mismatches.append(dict(csv=p.name,path=r['File Path'],field=key,
                                                   csv_value=r[source],actual=actual[key]))
    originals={Path(r['path']).stem:r for r in rows if r['family']=='original'}
    pairs=[]
    pairing={}
    for family in FAMILIES:
        rs=[r for r in rows if r['family']==family and not r['error']]
        for r in rs:
            target,donor=Path(r['path']).stem.split('_')
            orig=originals.get(target)
            if not orig or orig['error']:
                pairs.append(dict(fake=r['path'],target_id=target,missing_reference=True));continue
            pairs.append(dict(family=family,fake=r['path'],real=orig['path'],target_id=target,donor_id=donor,
                fake_frames=r['frames'],real_frames=orig['frames'],
                frame_delta=r['frames']-orig['frames'],fps_match=abs(r['fps']-orig['fps'])<1e-6,
                resolution_match=(r['width'],r['height'])==(orig['width'],orig['height']),
                fake_size=[r['width'],r['height']],real_size=[orig['width'],orig['height']],
                duration_delta_s=(r['duration_s'] or 0)-(orig['duration_s'] or 0),
                alignment_verified=False))
        ps=[p for p in pairs if p.get('family')==family]
        pairing[family]=dict(pairs=len(ps),equal_frames=sum(p['frame_delta']==0 for p in ps),
            fake_shorter=sum(p['frame_delta']<0 for p in ps),fake_longer=sum(p['frame_delta']>0 for p in ps),
            fps_mismatches=sum(not p['fps_match'] for p in ps),
            resolution_mismatches=sum(not p['resolution_match'] for p in ps),
            duration_delta_s=stats(p['duration_delta_s'] for p in ps))
    result=dict(folders=summary,pairing=pairing,csv_header_mismatches=csv_mismatches,
        total_files=len(paths),errors=sum(bool(r['error']) for r in rows),
        note='Container/header counts and first-frame decode for all files; not a full-frame integrity check.')
    (out/'metadata_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'pair_properties.json').write_text(json.dumps(pairs,indent=2)+'\n')
    return rows,pairs


def decode_small(path):
    thumbs=[];times=[];changes=[]
    with av.open(str(path)) as c:
        st=c.streams.video[0];st.codec_context.thread_count=1
        for frame in c.decode(st):
            # Full-image low-resolution screening, not face alignment.
            im=frame.reformat(width=64,height=48,format='gray').to_ndarray()
            thumbs.append(im)
            times.append(float(frame.time) if frame.time is not None else None)
    if not thumbs:
        raise ValueError('No decoded frames')
    return np.stack(thumbs),times


def pair_sample(root,out,pair):
    fake,ft=decode_small(root/pair['fake']);real,rt=decode_small(root/pair['real'])
    n=min(len(fake),len(real))
    # Borders reduce influence of the edited face, but remain only a heuristic.
    mask=np.ones((48,64),dtype=bool);mask[10:40,16:48]=False
    a=fake[:,mask].astype(np.float32);b=real[:,mask].astype(np.float32)
    diagonal=np.mean(np.abs(a[:n]-b[:n]),axis=1)
    sample_indices=np.unique(np.linspace(0,len(fake)-1,min(15,len(fake))).round().astype(int))
    nearest=[]
    for i in sample_indices:
        dist=np.mean(np.abs(b-a[i]),axis=1)
        j=int(dist.argmin())
        nearby=dist[max(0,j-2):min(len(dist),j+3)].copy()
        away=dist.copy();away[max(0,j-2):min(len(dist),j+3)]=np.inf
        margin=float(away.min()-dist[j]) if np.isfinite(away).any() else 0.
        nearest.append(dict(fake_index=int(i),nearest_real_index=j,mae=float(dist[j]),
                            same_index_mae=float(dist[i]) if i<len(dist) else None,
                            margin_to_nonlocal_match=margin))
    def time_stats(ts):
        vals=[t for t in ts if t is not None]
        diffs=np.diff(vals)
        return dict(frames=len(ts),timestamp_intervals_s=stats(diffs),
                    strictly_increasing=bool(np.all(diffs>0)),
                    distinct_intervals_rounded_us=dict(collections.Counter(str(int(round(x*1e6))) for x in diffs)))
    result=dict(**pair,decoded_fake_frames=len(fake),decoded_real_frames=len(real),
        fake_timing=time_stats(ft),real_timing=time_stats(rt),
        same_index_border_mae=stats(diagonal),nearest_matches=nearest,
        original_border_motion_mae=stats(np.abs(np.diff(b,axis=0)).mean(axis=1)),
        header_matches_decoded=(len(fake)==pair['fake_frames'] and len(real)==pair['real_frames']))
    return result


def sample_audit(root,out,pairs):
    selected=[]
    for family in FAMILIES:
        ps=[p for p in pairs if p.get('family')==family]
        # Deterministic, stratified diagnostics, not a representative accuracy estimate.
        strata=[('equal_same_size',lambda p:p['frame_delta']==0 and p['resolution_match']),
                ('shorter',lambda p:p['frame_delta']<0 and p['resolution_match']),
                ('longer',lambda p:p['frame_delta']>0 and p['resolution_match']),
                ('different_size',lambda p:not p['resolution_match'])]
        for name,predicate in strata:
            eligible=[p for p in ps if predicate(p)]
            # Shorter videos limit audit decoding cost. Spread samples by target ID.
            eligible=[p for p in eligible if max(p['fake_frames'],p['real_frames'])<=1000]
            for p in eligible[::max(1,len(eligible)//2)][:2]:
                if not any(q['fake']==p['fake'] for q in selected):
                    selected.append(dict(p,stratum=name))
    results=[]
    for p in selected:
        try:
            result=pair_sample(root,out,p)
        except Exception as e:
            result=dict(**p,error=str(e))
        results.append(result)
        print(f"Decoded pair {len(results)}/{len(selected)}: {p['fake']}",flush=True)
        (out/'sample_alignment.json').write_text(json.dumps(results,indent=2)+'\n')
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--reuse-metadata',action='store_true')
    args=parser.parse_args()
    root=Path(args.root).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if args.reuse_metadata:
        pairs=json.loads((out/'pair_properties.json').read_text())
    else:
        _,pairs=metadata_audit(root,out,args.workers)
    sample_audit(root,out,pairs)
    print('Audit complete:',out,flush=True)


if __name__=='__main__':
    main()
