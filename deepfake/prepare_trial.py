"""Prepare a small single-method trial, holding out target/donor components."""
import argparse
import json
import random
from pathlib import Path

import cv2
import torch

from .data import PreparedClipDataset, audit_splits
from .preprocess import prepare_pair, validate_output, CONVENTION


def split_candidates(rows, seed=42):
    parent = {}
    def find(x):
        parent.setdefault(x,x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for r in rows:
        source, donor = Path(r['video_path']).stem.split('_')
        parent[find(source)] = find(donor)
    groups = {}
    for r in rows:
        source = Path(r['video_path']).stem.split('_')[0]
        groups.setdefault(find(source),[]).append(r)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError('Need at least two disconnected source/donor groups')
    rng = random.Random(seed)
    rng.shuffle(keys)
    held_out = set(keys[:max(1,len(keys)//5)])
    pools = {'train':[], 'val':[]}
    for key in keys:
        pools['val' if key in held_out else 'train'].extend(groups[key])
    for pool in pools.values():
        rng.shuffle(pool)
    return pools


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',default='Deepfakes')
    p.add_argument('--train-pairs',type=int,default=100)
    p.add_argument('--eval-pairs',type=int,default=20)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--frames',type=int,default=17)
    p.add_argument('--start-frame',type=int,default=32)
    p.add_argument('--frame-stride',type=int,default=2)
    p.add_argument('--size',type=int,default=256)
    p.add_argument('--face-margin',type=float,default=1.4)
    p.add_argument('--max-background-mae',type=float,default=.04)
    p.add_argument('--max-background-p95',type=float,default=.12)
    a=p.parse_args()
    if min(a.train_pairs,a.eval_pairs,a.frames,a.frame_stride)<1 or a.start_frame<0 or a.size<32 or a.size%32:
        p.error('Invalid counts, frame sampling, or size')
    root,out=a.raw_root.resolve(),a.output.resolve()
    validate_output(root,out)
    candidates=[json.loads(s) for s in a.pairs.read_text().splitlines() if s.strip()]
    candidates=[r for r in candidates if Path(r['video_path']).parent.name==a.family]
    pools=split_candidates(candidates,a.seed)
    torch.set_num_threads(4);cv2.setNumThreads(4)
    out.mkdir(parents=True)
    for name in ('clips','reports','previews'):
        (out/name).mkdir()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    (out/'config.json').write_text(json.dumps(dict(config,map_convention=CONVENTION,
        split_policy='20% of target/donor connected components reserved for evaluation; unused candidates remain reserved',
        evaluation='val: monitored holdout, not an untouched final test set'),indent=2)+'\n')
    (out/'candidate_split.json').write_text(json.dumps(pools,indent=2)+'\n')
    reports,manifest=[],[]
    counts={}
    for split,goal in [('val',a.eval_pairs),('train',a.train_pairs)]:
        a.split=split
        accepted=0
        for row in pools[split]:
            if accepted==goal:
                break
            try:
                samples,report=prepare_pair(root,out,row,a,len(reports))
            except (ValueError,OSError,cv2.error) as exc:
                samples=[]
                report=dict(video_path=row['video_path'],status='rejected',error=str(exc))
            report['split']=split
            reports.append(report)
            manifest.extend(samples)
            accepted+=bool(samples)
            (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
            (out/'pair_reports.json').write_text(json.dumps(reports,indent=2)+'\n')
            print(f"{split}: {accepted}/{goal} accepted; {row['video_path']}: {report['status']}",flush=True)
        counts[split]=accepted
    audit_splits(out/'manifest.jsonl')
    for split,count in counts.items():
        if count:
            ds=PreparedClipDataset(out/'manifest.jsonl',split)
            for i in range(len(ds)):
                ds[i]
    complete=counts=={'val':a.eval_pairs,'train':a.train_pairs}
    summary=dict(complete=complete,accepted_pairs=counts,clips={k:2*v for k,v in counts.items()},
                 attempted_pairs=len(reports),rejected_pairs=sum(r['status']!='accepted' for r in reports),
                 source_and_donor_disjoint=True,loader_verified=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if not complete:
        raise SystemExit('Insufficient accepted pairs; do not launch training with this incomplete trial')


if __name__=='__main__':
    main()
