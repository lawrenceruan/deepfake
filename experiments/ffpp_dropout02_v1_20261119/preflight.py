import json,sys,torch
from pathlib import Path
from deepfake.model import DetectorHeads
from deepfake.config import DetectorConfig
p=Path(sys.argv[1]);baseline=json.loads((p/'source/configs/wan_only.json').read_text());cfg=json.loads((p/'dropout02.json').read_text());assert {k:(baseline[k],v) for k,v in cfg.items() if baseline[k]!=v}=={'dropout':(0.0,0.2)}
torch.set_num_threads(4)
torch.manual_seed(42);a=DetectorHeads(DetectorConfig(**baseline));torch.manual_seed(42);b=DetectorHeads(DetectorConfig(**cfg))
assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
x=[torch.randn(1,3072,2,2,2) for _ in range(3)];valid=torch.ones(1,2,dtype=torch.bool)
a.eval();b.eval()
with torch.no_grad():
 ra=a(x,valid,(5,64,64));rb=b(x,valid,(5,64,64))
 assert all(torch.equal(ra[k],rb[k]) for k in ra)
 b.train();r1=b(x,valid,(5,64,64));r2=b(x,valid,(5,64,64));assert not torch.equal(r1['visual_logit'],r2['visual_logit'])
report={'only_config_change':'dropout 0.0 -> 0.2','initial_head_weights_identical':True,'eval_outputs_identical':True,'training_dropout_active':True,'dropout_locations':[name for name,m in b.named_modules() if isinstance(m,torch.nn.Dropout)],'attention_dropout':[m.dropout for m in b.modules() if isinstance(m,torch.nn.MultiheadAttention)]}
(p/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
# Check the adapter's feature representation against one saved baseline prediction.
root=p.parents[2];cache=root/'datasets/features/ffpp_wan_features_fp32_v1_20261118';run=root/'ours/runs/ffpp_joint_wan_only_v1_20261118_786090'
record=next(r for line in (cache/'manifest.jsonl').read_text().splitlines() for r in [json.loads(line)] if r['split']=='val')
x=torch.load(cache/record['feature_path'],weights_only=True,map_location='cpu');state=torch.load(run/'best.pt',weights_only=True,map_location='cpu');b.load_state_dict(state['heads']);b.cuda().eval()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):result=b([g[None].cuda() for g in x['grids']],x['latent_validity'][None].cuda(),x['output_size'],x['frame_validity'][None].cuda())
expected=next(r for r in json.loads((run/'verified_predictions.json').read_text()) if r['sample_id']==record['sample_id'])
score=float(result['visual_logit'].float().sigmoid()[0]);assert abs(score-expected['fake_probability'])<1e-5
report['cached_baseline_probability_verified']=True;report['sample_id']=record['sample_id'];report['probability_absolute_error']=abs(score-expected['fake_probability']);(p/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print('Cached feature baseline reproduction passed',flush=True)
