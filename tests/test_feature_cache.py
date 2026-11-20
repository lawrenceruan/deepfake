import json
import pytest
import torch
from deepfake.feature_cache import (CachedWanFeatureDataset, load_features,
    save_features, sha256_file, write_json)


def example():
    row = dict(sample_id='fake_001', split='train', generator_family='Deepfakes',
               visual_label=1, tensor_path='/untouched/prepared.pt')
    payload = dict(sample_id=row['sample_id'], extraction_id='contract-a',
        feature_layers=[9,19,29], grids=[torch.randn(8,2,4,4) for _ in range(3)],
        latent_validity=torch.tensor([True,True]),
        frame_validity=torch.tensor([True,True,True,True,True,False]), output_size=[6,32,32])
    return row,payload


def test_lossless_roundtrip_identity_and_corruption(tmp_path):
    row,payload=example()
    record=save_features(tmp_path,row,payload,'source-hash')
    restored=load_features(tmp_path,record,'contract-a')
    assert all(torch.equal(a,b) for a,b in zip(payload['grids'],restored['grids']))
    assert restored['frame_validity'].tolist()==payload['frame_validity'].tolist()
    with pytest.raises(ValueError,match='configuration mismatch'):
        load_features(tmp_path,record,'wrong-config')
    with pytest.raises(FileExistsError):save_features(tmp_path,row,payload,'source-hash')
    path=tmp_path/record['feature_path']
    with path.open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):load_features(tmp_path,record,'contract-a')


def test_completed_loader_checks_split_manifest_and_config(tmp_path):
    row,payload=example();record=save_features(tmp_path,row,payload,'source-hash')
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(json.dumps(record)+'\n')
    with pytest.raises(FileNotFoundError):CachedWanFeatureDataset(tmp_path,'train')
    write_json(tmp_path/'complete.json',dict(complete=True,clips=1,
        extraction_id='contract-a',manifest_sha256=sha256_file(manifest)))
    ds=CachedWanFeatureDataset(tmp_path,'train','contract-a')
    assert len(ds)==1 and ds[0]['metadata']['tensor_path']=='/untouched/prepared.pt'
    with pytest.raises(ValueError,match='No cached samples'):CachedWanFeatureDataset(tmp_path,'val')
    with pytest.raises(ValueError,match='incompatible'):CachedWanFeatureDataset(tmp_path,'train','other')
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError,match='checksum'):CachedWanFeatureDataset(tmp_path,'train')


def test_reject_invalid_feature_dtype_and_external_path(tmp_path):
    row,payload=example();payload['grids'][0]=payload['grids'][0].bfloat16()
    with pytest.raises(ValueError,match='FP32'):save_features(tmp_path,row,payload,'x')
    with pytest.raises(ValueError,match='outside cache'):
        load_features(tmp_path,dict(feature_path='../escape.pt'),'contract-a')
