"""Use the original trainer and initialization, with verified disk features."""
import hashlib,json,sys
from pathlib import Path
import torch
import deepfake.train as training
ROOT=Path('/project/llmsvgen/junhong/Video_Deepfake')
CACHE=ROOT/'datasets/features/ffpp_wan_features_fp32_v1_20261118'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def arg(name):return Path(sys.argv[sys.argv.index(name)+1]).resolve()
manifest=arg('--manifest');cfg=json.loads(arg('--config').read_text());meta=json.loads((CACHE/'metadata.json').read_text());complete=json.loads((CACHE/'complete.json').read_text())
assert complete['complete'] and sha(CACHE/'manifest.jsonl')==complete['manifest_sha256']
assert sha(manifest)==meta['contract']['source_manifest_sha256']
for k,v in meta['contract']['extraction'].items():assert cfg[k]==v,(k,cfg[k],v)
assert sha(arg('--context'))==meta['contract']['context_sha256']
rows=[json.loads(x) for x in (CACHE/'manifest.jsonl').read_text().splitlines()];records={(r['split'],r['sample_id']):r for r in rows}
assert len(records)==36004
source=Path(training.__file__).parent.parent
protected=[manifest,CACHE/'manifest.jsonl',ROOT/'ours/runs/ffpp_joint_wan_only_v1_20261118_786090/best.pt',ROOT/'ours/runs/ffpp_joint_wan_only_v1_20261118_786090/last.pt']+list((source/'deepfake').glob('*.py'))
before={str(p):sha(p) for p in protected}

def cached_forward(model,batch,feature_cache=None):
 assert feature_cache is not None and len(batch['video'])==1
 row=batch['metadata'][0];key=(row['split'],row['sample_id'])
 if key not in feature_cache:
  record=records[key];path=(CACHE/record['feature_path']).resolve();assert path.is_relative_to(CACHE)
  assert sha(path)==record['feature_sha256']
  payload=torch.load(path,weights_only=True,map_location='cpu')
  assert payload['sample_id']==row['sample_id'] and payload['extraction_id']==complete['extraction_id']
  assert payload['feature_layers']==cfg['feature_layers']
  assert torch.equal(payload['frame_validity'],batch['frame_validity'][0].cpu())
  assert list(payload['output_size'])==[batch['video'].shape[1],*batch['video'].shape[-2:]]
  for g in payload['grids']:assert g.dtype==torch.float32 and torch.isfinite(g).all()
  feature_cache[key]=([g[None] for g in payload['grids']],payload['latent_validity'][None])
  if len(feature_cache)==1:print({'disk_feature_cache_verified':True,'extraction_id':complete['extraction_id'],'dropout':cfg['dropout']},flush=True)
 grids,valid=feature_cache[key];device=batch['video'].device
 return model.heads([g.to(device) for g in grids],valid.to(device),(batch['video'].shape[1],*batch['video'].shape[-2:]),batch['frame_validity'])
training.trial_forward=cached_forward
training.main()
after={str(p):sha(p) for p in protected};assert before==after
(arg('--output')/'preservation_check.json').write_text(json.dumps({'match':True,'sha256':after},indent=2)+'\n')
