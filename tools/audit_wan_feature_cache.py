"""Audit a completed feature cache and bundle metadata without copying tensors."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tarfile
from deepfake.data import audit_splits
from deepfake.feature_cache import (CachedWanFeatureDataset, load_features,
                                   sha256_file, signature, write_json)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();root=a.cache.resolve();exp=a.experiment.resolve()
    meta=json.loads((root/'metadata.json').read_text())
    done=json.loads((root/'complete.json').read_text())
    assert done['complete'] and done['scope']=='all_active_train_and_validation'
    assert signature(meta['contract'])==meta['extraction_id']==done['extraction_id']
    assert sha256_file(root/'manifest.jsonl')==done['manifest_sha256']
    assert sha256_file(root/'source_manifest.jsonl')==meta['contract']['source_manifest_sha256']
    assert sha256_file(meta['contract']['source_manifest'])==meta['contract']['source_manifest_sha256']
    originals={r['sample_id']:r for line in (root/'source_manifest.jsonl').read_text().splitlines()
               for r in [json.loads(line)]}
    rows=[json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines()]
    assert len(rows)==done['clips']==len(originals)==36004
    assert len({r['sample_id'] for r in rows})==len(rows)
    size=0;selected={}
    for row in rows:
        assert all(row[k]==v for k,v in originals[row['sample_id']].items())
        assert row['extraction_id']==done['extraction_id'] and row['feature_dtype']=='float32'
        assert len(row['feature_shapes'])==3
        assert all(s[0]==3072 and s[2:]==[8,8] for s in row['feature_shapes'])
        size+=(root/row['feature_path']).stat().st_size
        key=(row['split'],row['generator_family'],row['visual_label'],
             'full' if row['segment']['valid_frames']==25 else 'short')
        selected.setdefault(key,row)
    audit_splits(root/'manifest.jsonl')
    verified=[]
    for row in selected.values():
        sample=load_features(root,row,done['extraction_id'])
        assert int(sample['frame_validity'].sum())==row['segment']['valid_frames']
        assert sample['output_size']==[25,256,256]
        assert sample['feature_layers']==[9,19,29]
        assert sha256_file(row['tensor_path'])==row['source_tensor_sha256']
        verified.append(row['sample_id'])
    for split in ('train','val'):
        ds=CachedWanFeatureDataset(root,split,done['extraction_id']);ds[0]
    report=dict(complete=True,cache=str(root),clips=len(rows),
        splits=dict(Counter(r['split'] for r in rows)),
        methods=dict(Counter(r['generator_family'] for r in rows)),
        feature_file_bytes=size,feature_file_gib=size/2**30,
        extraction_id=done['extraction_id'],source_manifest_unchanged=True,
        original_metadata_and_splits_preserved=True,split_audit_passed=True,
        all_feature_files_present=True,
        exact_serialization_verified_for_every_entry_by_exporter=done['serialized_features_exactly_verified'],
        post_export_payload_and_input_checksum_samples=verified,
        repeat_extraction_checks=done['repeat_checks'],cache_loader_verified=True,
        current_training_modified=False)
    write_json(exp/'cache_audit.json',report)
    bundle=exp/'cache_metadata.tar.gz'
    if bundle.exists():raise FileExistsError('Metadata bundle already exists; preserve it')
    with tarfile.open(bundle,'w:gz',compresslevel=3) as t:
        for name in ('metadata.json','complete.json','manifest.jsonl','preprocessing_config.json'):
            t.add(root/name,arcname='cache_metadata/'+name)
        t.add(exp/'cache_audit.json',arcname='cache_metadata/cache_audit.json')
        t.add(exp/'usage.txt',arcname='cache_metadata/usage.txt')
    result=dict(report=report,bundle=str(bundle),bundle_bytes=bundle.stat().st_size,
                bundle_sha256=sha256_file(bundle))
    write_json(exp/'delivery.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
