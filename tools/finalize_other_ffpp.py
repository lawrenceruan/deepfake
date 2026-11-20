"""Audit finished FF++ methods and build a joint metadata-only loader manifest."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from deepfake.data import PreparedClipDataset, audit_splits
from deepfake.preprocessing_config import PreprocessingConfig
from deepfake.prepare_full_segments import write_json, write_rows


FAMILIES=('Face2Face','FaceSwap','FaceShifter','NeuralTextures')
EXPERIMENT='ffpp_other_paired_3s8fps_v1_20261118'
BASE='deepfakes_full_3s8fps_eval50_v1_20261118'
COMBINED='ffpp_all_paired_3s8fps_sharedholdout_v1_20261118'


def load_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def recording_ids(rows):
    return {x for r in rows for x in (r['source_video_id'],r['donor_sequence_id'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    exp=root/'ours/experiments'/EXPERIMENT
    sources={'Deepfakes':root/'datasets/preprocessed'/BASE}
    sources.update({f:root/'datasets/preprocessed'/f'{f.lower()}_full_3s8fps_sharedholdout_v1_20261118' for f in FAMILIES})
    prepared={};fingerprints=set();holdout=None
    for family,folder in sources.items():
        summary=json.loads((folder/'summary.json').read_text())
        assert summary['complete'] and summary['candidate_pairs']==1000
        assert summary['raw_sha256_verified_before_and_after'] and summary['loader_verified'] and summary['split_audit_passed']
        cfg=PreprocessingConfig.load(folder/'preprocessing_config.json');fingerprints.add(cfg.fingerprint)
        assert summary['preprocessing_config_id']==cfg.fingerprint
        selected=set(summary['split']['selected_components'])
        if holdout is None:holdout=selected
        assert selected==holdout and len(selected)==50
        active=load_rows(folder/'manifest.jsonl');reserved=load_rows(folder/'reserved_holdout_relatives.jsonl')
        all_rows=load_rows(folder/'all_prepared_manifest.jsonl')
        assert {r['sample_id'] for r in all_rows}=={r['sample_id'] for r in active+reserved}
        assert len(all_rows)==len(active)+len(reserved)==summary['prepared_clips']
        assert all(r['recording_component'] not in holdout for r in active if r['split']=='train')
        assert all(r['recording_component'] in holdout for r in reserved+[r for r in active if r['split']=='val'])
        audit_splits(folder/'manifest.jsonl')
        prepared[family]=(folder,summary,active,reserved,all_rows)
    assert len(fingerprints)==1
    combined=root/'datasets/manifests'/COMBINED
    if combined.exists():raise ValueError('Combined manifest folder already exists; preserve it and use a new version')
    combined.mkdir(parents=True)
    merged={key:[] for key in ['active','reserved','all']};report={}
    metadata=exp/'completed_metadata';metadata.mkdir(exist_ok=True)
    for family,(folder,summary,active,reserved,all_rows) in prepared.items():
        def convert(row):
            r=dict(row,origin_dataset=folder.name)
            for key in ['tensor_path','geometry_report','preprocessing_config_path']:
                if r.get(key) and not Path(r[key]).is_absolute():r[key]=str(folder/r[key])
            return r
        for key,rs in [('active',active),('reserved',reserved),('all',all_rows)]:merged[key].extend(convert(r) for r in rs)
        counts={}
        for split in ['train','val']:
            rs=[r for r in active if r['split']==split]
            counts[split]=dict(clips=len(rs),labels=dict(Counter(str(r['visual_label']) for r in rs)),
                parent_video_files=len({r['raw_video_path'] for r in rs}),recording_groups=len({r['recording_component'] for r in rs}),
                valid_frames=sum(r['segment']['valid_frames'] for r in rs))
        report[family]=dict(candidate_pairs=1000,accepted_source_pairs=summary['pairs_with_accepted_segments'],
            accepted_segment_pairs=summary['accepted_segment_pairs'],prepared_clips=summary['prepared_clips'],
            reserved_clips=len(reserved),splits=counts,rejection_reasons=summary['rejection_reasons'],
            segment_status_counts=summary['segment_status_counts'],
            missing_evaluation_components=summary['split'].get('missing_evaluation_components',[]),
            path=str(folder),manifest_sha256=hashlib.sha256((folder/'manifest.jsonl').read_bytes()).hexdigest())
        dest=metadata/family;dest.mkdir(exist_ok=True)
        for name in ['summary.json','preprocessing_config.json','manifest.jsonl','reserved_holdout_relatives.jsonl']:
            shutil.copy2(folder/name,dest/name)
        if family in FAMILIES:
            # Prefer an accepted center-cropped validation example when available.
            candidates=[r for r in active if r['split']=='val' and r['visual_label']==1]
            picked=None
            for row in candidates:
                geo=folder/row['geometry_report'];details=json.loads(geo.read_text())
                if picked is None or details['real_center_crop_xyxy'][0]>0:picked=(row,geo,details)
                if details['real_center_crop_xyxy'][0]>0:break
            if picked:
                row,geo,details=picked
                shutil.copy2(geo,dest/'preview_geometry.json')
                shutil.copy2(geo.parent.parent/details['preview'],dest/'preprocessing_preview.png')
                write_json(dest/'preview_selection.json',dict(sample_id=row['sample_id'],split='val',
                    center_crop_xyxy=details['real_center_crop_xyxy'],selection='first accepted center-cropped val fake if available, otherwise first val fake'))
    assert len({r['sample_id'] for r in merged['all']})==len(merged['all'])
    train=[r for r in merged['active'] if r['split']=='train'];val=[r for r in merged['active'] if r['split']=='val']
    assert not recording_ids(train)&recording_ids(val+merged['reserved'])
    write_rows(combined/'manifest.jsonl',merged['active'])
    write_rows(combined/'reserved_holdout_relatives.jsonl',merged['reserved'])
    write_rows(combined/'all_prepared_manifest.jsonl',merged['all'])
    shutil.copy2(sources['Deepfakes']/'preprocessing_config.json',combined/'preprocessing_config.json')
    audit_splits(combined/'manifest.jsonl')
    # Exercise actual loader path resolution for both labels and splits in every method.
    checked=[]
    for split in ['train','val']:
        ds=PreparedClipDataset(combined/'manifest.jsonl',split,next(iter(fingerprints)))
        seen=set()
        for i,row in enumerate(ds.rows):
            key=(row['generator_family'],row['visual_label'])
            if key not in seen:
                item=ds[i];assert item['video'].shape==(25,3,256,256)
                seen.add(key);checked.append(row['sample_id'])
        assert len(seen)==10
    summary=dict(complete=True,metadata_only=True,preprocessing_config_id=next(iter(fingerprints)),
        active_clips=dict(Counter(r['split'] for r in merged['active'])),reserved_clips=len(merged['reserved']),
        families=report,heldout_recording_groups=50,validation_entries=len(val),
        validation_unique_parent_paths=len({r['raw_video_path'] for r in val}),
        shared_train_heldout_recordings=[],loader_checked_samples=checked,
        combined_manifest=str(combined/'manifest.jsonl'),
        real_crop_policy='Per-method original crops retained as existing dataset entries; no new tensors copied. Some real entries across methods may be identical. Choose sampling/deduplication policy when training.',
        evaluation_note='Same 50 recording groups across methods; validation entries are not independent recording groups.',
        excluded={'DeepFakeDetection':'Pending paired actor-original videos, as requested.'})
    write_json(combined/'summary.json',summary);write_json(metadata/'preprocessing_report.json',summary)
    text='Preprocessing completed for the four additional paired FF++ c23 sets.\n\n'
    for family,r in report.items():
        text+=f"{family}: {r['accepted_source_pairs']}/1000 pairs produced accepted clips; {r['prepared_clips']} prepared clips; train {r['splits']['train']['clips']}; validation {r['splits']['val']['clips']}; reserved {r['reserved_clips']}.\n"
    text+='\nSame 3-second / 8-FPS / 25-frame / 256-square preprocessing and DSSIM convention as Deepfakes. Short tails are padded and masked. All source checksums, per-method loader checks and split audits passed.\n'
    text+='\nThe same 50 target/donor recording groups are held out across all five methods. No training recording overlaps any validation or reserved group. Missing accepted evaluation groups are reported without replacing them with training groups.\n'
    text+='\nCombined manifest for later training (metadata only, absolute tensor paths):\n'+str(combined/'manifest.jsonl')+'\n'
    text+='\n'+summary['real_crop_policy']+'\n'+summary['evaluation_note']+'\n'
    text+='\nDeepFakeDetection remains pending because the Kaggle mirror lacks its paired actor-original videos. Raw files and all earlier Deepfakes datasets/checkpoints are preserved. No new detector training was launched.\n'
    (metadata/'preprocessing_report.txt').write_text(text)
    for name in ['manifest.jsonl','summary.json','preprocessing_config.json']:
        target=metadata/'combined';target.mkdir(exist_ok=True);shutil.copy2(combined/name,target/name)
    bundle=exp/'completed_preprocessing_metadata.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=3) as t:t.add(metadata,arcname='completed_metadata')
    print(json.dumps(dict(complete=True,families={f:r['splits'] for f,r in report.items()},
                         combined_manifest=str(combined/'manifest.jsonl'),bundle=str(bundle),
                         bundle_bytes=bundle.stat().st_size,bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest())),flush=True)


if __name__=='__main__':main()
