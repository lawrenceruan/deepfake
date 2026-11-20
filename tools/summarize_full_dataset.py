"""Compact data-accounting report for the finished versioned dataset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('dataset',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
summary=json.loads((a.dataset/'summary.json').read_text())
assert summary['complete']
manifest=a.dataset/'manifest.jsonl'
rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
reserved=[json.loads(s) for s in (a.dataset/'reserved_holdout_relatives.jsonl').read_text().splitlines() if s.strip()]
report=dict(candidate_pairs=summary['candidate_pairs'],accepted_source_pairs=summary['pairs_with_accepted_segments'],
    accepted_segment_pairs=summary['accepted_segment_pairs'],prepared_clips=summary['prepared_clips'],
    reserved_holdout_relative_clips=len(reserved),segment_status_counts=summary['segment_status_counts'],
    rejection_reasons=summary['rejection_reasons'],preprocessing_config_id=summary['preprocessing_config_id'],
    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),splits={})
for split in ('train','val'):
    selected=[r for r in rows if r['split']==split]
    unique_frames={(r['raw_video_path'],i) for r in selected for i in r['frame_indices']}
    report['splits'][split]=dict(clips=len(selected),labels=dict(Counter(str(r['visual_label']) for r in selected)),
        source_recordings=len({r['source_video_id'] for r in selected}),
        parent_video_files=len({r['raw_video_path'] for r in selected}),
        recording_components=len({r['recording_component'] for r in selected}),
        valid_frame_samples=sum(r['segment']['valid_frames'] for r in selected),
        unique_parent_frame_indices=len(unique_frames),
        short_tail_clips=sum(r['segment']['valid_frames']<r['segment']['maximum_frames'] for r in selected))
train={x for r in rows if r['split']=='train' for x in (r['source_video_id'],r['donor_sequence_id'])}
val={x for r in rows if r['split']=='val' for x in (r['source_video_id'],r['donor_sequence_id'])}
assert not train&val
report['shared_train_eval_recordings']=sorted(train&val)
report['heldout_original_recordings']=len(val)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
