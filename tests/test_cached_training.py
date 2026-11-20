import json
import pytest
import torch
from deepfake.cached_training import DiskWanFeatureCache
from deepfake.config import DetectorConfig
from deepfake.feature_cache import save_features, sha256_file, write_json


def cache_fixture(tmp_path):
    root=tmp_path/'cache';root.mkdir()
    context=tmp_path/'context.pt';torch.save(torch.ones(2),context)
    cfg=DetectorConfig(feature_layers=(0,),backbone_dim=4)
    rows=[];records=[]
    for split in ('train','val'):
        source=tmp_path/(split+'.pt');torch.save({'video':torch.rand(1,3,32,32)},source)
        row=dict(sample_id=split,split=split,tensor_path=str(source))
        payload=dict(sample_id=split,extraction_id='fixed',feature_layers=[0],
                     grids=[torch.rand(4,1,1,1)],latent_validity=torch.ones(1,dtype=torch.bool),
                     frame_validity=torch.ones(1,dtype=torch.bool),output_size=[1,32,32])
        records.append(save_features(root,row,payload,sha256_file(source)));rows.append(row)
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cached_manifest=root/'manifest.jsonl';cached_manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    write_json(root/'complete.json',dict(complete=True,clips=2,extraction_id='fixed',manifest_sha256=sha256_file(cached_manifest)))
    extraction={k:cfg.to_dict()[k] for k in ('feature_layers','backbone_dim','probe_sigma','probe_seed','num_train_timesteps')}
    write_json(root/'metadata.json',dict(extraction_id='fixed',contract=dict(extraction=extraction,
        context_sha256=sha256_file(context),source_manifest_sha256=sha256_file(manifest))))
    batch=dict(video=torch.rand(1,1,3,32,32),frame_validity=torch.ones(1,1,dtype=torch.bool),metadata=[rows[0]])
    return root,manifest,context,cfg,batch


def test_disk_cache_read_only_and_rejects_changed_rgb_inputs(tmp_path):
    root,manifest,context,cfg,batch=cache_fixture(tmp_path)
    before={p:sha256_file(p) for p in tmp_path.rglob('*') if p.is_file()}
    cache=DiskWanFeatureCache(root,manifest,context,cfg)
    grids,valid=cache.get_batch(batch)
    assert grids[0].shape==(1,4,1,1,1) and valid.tolist()==[[True]]
    assert before=={p:sha256_file(p) for p in before}
    with open(batch['metadata'][0]['tensor_path'],'ab') as handle:handle.write(b'changed')
    with pytest.raises(ValueError,match='changed during'):cache.get_batch(batch)


def test_cache_rejects_contract_and_geometry_mismatches(tmp_path):
    root,manifest,context,cfg,batch=cache_fixture(tmp_path)
    cache=DiskWanFeatureCache(root,manifest,context,cfg)
    batch['video']=torch.rand(1,1,3,64,64)
    with pytest.raises(ValueError,match='geometry'):cache.get_batch(batch)
    context.write_bytes(b'changed')
    with pytest.raises(ValueError,match='context'):DiskWanFeatureCache(root,manifest,context,cfg)
