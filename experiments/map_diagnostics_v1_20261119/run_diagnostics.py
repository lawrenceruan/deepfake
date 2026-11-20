"""Isolated diagnostics using unchanged frozen training source and cached features."""
import argparse, json, hashlib, random, sys
from pathlib import Path
import numpy as np
import torch
from deepfake.model import DetectorHeads
from deepfake.config import DetectorConfig
from deepfake.data import PreparedClipDataset

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
 run=a.root/'ours/runs/ffpp_joint_wan_only_v1_20261118_786090';cache=a.root/'datasets/features/ffpp_wan_features_fp32_v1_20261118'
 state=torch.load(run/'best.pt',weights_only=True,map_location='cpu');cfg=DetectorConfig(**state['config']);manifest=Path(state['manifest'])
 protected=[run/'best.pt',run/'last.pt',manifest,cache/'manifest.jsonl']
 protected+=list(Path(sys.modules[DetectorHeads.__module__].__file__).parent.glob('*.py'))
 before={str(p):sha(p) for p in protected};save(a.out/'protected_checksums_before.json',before)
 rows=[json.loads(x) for x in manifest.read_text().splitlines()];records={r['sample_id']:r for x in (cache/'manifest.jsonl').read_text().splitlines() for r in [json.loads(x)]}
 summary=json.loads((cache/'complete.json').read_text());assert summary['complete'] and sha(cache/'manifest.jsonl')==summary['manifest_sha256']
 families=sorted({r['generator_family'] for r in rows});train=[r for r in rows if r['split']=='train' and r['visual_label']==1]
 sums={f:torch.zeros(1,256,256,dtype=torch.float64) for f in families};counts={f:torch.zeros(1,256,256,dtype=torch.float64) for f in families};middle={f:[] for f in families};input_stats={}
 def touch(path):
  path=Path(path);s=path.stat();input_stats[str(path)]=(s.st_size,s.st_mtime_ns)
 def payload(row):
  path=manifest.parent/row['tensor_path'];touch(path)
  return torch.load(path,weights_only=True,map_location='cpu',mmap=True)
 mask0=None
 for i,r in enumerate(train):
  x=payload(r);n=int(x['frame_validity'].sum());y=x['map_target'][:n].float();m=x['map_validity'][:n].bool();f=r['generator_family']
  if mask0 is None:mask0=m[0].clone()
  assert torch.equal(m,mask0[None].expand_as(m))
  sums[f]+=(y*m).sum(0,dtype=torch.float64);counts[f]+=m.sum(0);middle[f].append(y[n//2][mask0].numpy().copy())
  if (i+1)%500==0: print('templates',i+1,len(train),flush=True)
 templates={f:{'mean':(sums[f]/counts[f].clamp_min(1)).float()} for f in families}
 for f in families:
  med=torch.zeros_like(mask0,dtype=torch.float32);mat=np.stack(middle[f]);med[mask0]=torch.from_numpy(np.median(np.ascontiguousarray(mat.T),axis=1,overwrite_input=True));templates[f]['median']=med;del mat;print('method median complete',f,flush=True)
 global_mean=(sum(sums.values())/sum(counts.values()).clamp_min(1)).float()
 mat=np.concatenate([np.stack(middle[f]) for f in families]);global_median=torch.zeros_like(global_mean);global_median[mask0]=torch.from_numpy(np.median(np.ascontiguousarray(mat.T),axis=1,overwrite_input=True));del mat,middle
 torch.save({'by_method':templates,'global_mean':global_mean,'global_median':global_median,'mask':mask0},a.out/'training_templates.pt')
 save(a.out/'template_training_ids.json',[r['sample_id'] for r in train]);print('templates complete',flush=True)
 model=DetectorHeads(cfg).cuda();model.load_state_dict(state['heads']);model.eval()
 ds=PreparedClipDataset(manifest,'val');verified={x['sample_id']:x for x in json.loads((run/'verified_predictions.json').read_text())}
 def feature(row):
  r=records[row['sample_id']];path=cache/r['feature_path'];touch(path);assert sha(path)==r['feature_sha256']
  x=torch.load(path,weights_only=True,map_location='cpu');assert x['sample_id']==row['sample_id'] and x['extraction_id']==summary['extraction_id'];return x
 def forward(x):
  with torch.autocast('cuda',dtype=torch.bfloat16):
   return model([g[None].cuda() for g in x['grids']],x['latent_validity'][None].cuda(),x['output_size'],x['frame_validity'][None].cuda())
 results=[];centers={f:[] for f in families};visuals=[]
 with torch.no_grad():
  for i in range(len(ds)):
   item=ds[i];r=item['metadata'];touch(manifest.parent/r['tensor_path']);x=feature(r);out=forward(x);pred=out['modification_map'][0].float().cpu();y=item['map_target'];m=item['map_validity'];prob=float(out['visual_logit'].float().sigmoid()[0]);err=float((pred-y).abs()[m].mean())
   assert abs(err-verified[r['sample_id']]['map_mae'])<1e-5 and abs(prob-verified[r['sample_id']]['fake_probability'])<1e-5
   if not r['visual_label']:continue
   f=r['generator_family'];n=int(item['frame_validity'].sum());k=n//2
   values={'model':pred,'zero':torch.zeros_like(pred),'global_mean':global_mean[None].expand_as(pred),'global_median':global_median[None].expand_as(pred),**{f'method_{key}':v[None].expand_as(pred) for key,v in templates[f].items()}}
   errors={key:float((v-y).abs()[m].sum()) for key,v in values.items()}
   results.append(dict(sample_id=r['sample_id'],method=f,pixels=int(m.sum()),errors=errors))
   centers[f].append((pred[k][mask0].numpy(),y[k][mask0].numpy()))
   if len(centers[f])<=3:visuals.append(dict(method=f,sample_id=r['sample_id'],video=item['video'][k],target=y[k],prediction=pred[k],template=templates[f]['median']))
   print('validation',i+1,len(ds),flush=True)
 report={'checkpoint_epoch':state['epoch']+1,'training_fake_clips':len(train),'evaluation_fake_clips':len(results),'verified_reproduction_all_250':True,'methods':{}}
 for f in families+['combined']:
  rs=results if f=='combined' else [r for r in results if r['method']==f];report['methods'][f]={key:sum(r['errors'][key] for r in rs)/sum(r['pixels'] for r in rs) for key in rs[0]['errors']}
 rng=np.random.default_rng(42);shuffle={}
 for f in families:
  cp=np.stack([r[0] for r in centers[f]]);ct=np.stack([r[1] for r in centers[f]]);pair=np.stack([np.abs(cp-y).mean(1) for y in ct]);ix=np.arange(len(cp));scores=[]
  for _ in range(2000):
   order=rng.permutation(len(ix))
   while np.any(order==ix):order=rng.permutation(len(ix))
   scores.append(float(pair[ix,order].mean()))
  shuffle[f]={'own_middle_frame_mae':float(np.diag(pair).mean()),'shuffled_middle_frame_mae_mean':float(np.mean(scores)),'shuffle_2_5_97_5_percentiles':np.percentile(scores,[2.5,97.5]).tolist(),'note':'Descriptive within-method derangements; middle frames only, shared recording groups, not an independent significance test.'}
 report['shuffle']=shuffle;save(a.out/'diagnosis.json',report);save(a.out/'per_clip.json',results);torch.save(visuals,a.out/'diagnostic_visuals.pt');print(json.dumps(report),flush=True)
 # Tiny-set memorization: existing architecture, pretrained initialization, map-only objective.
 candidates=sorted([r for r in rows if r['split']=='train'],key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest());selected=[];groups=set()
 for f in families:
  for label,needed in [(1,4),(0,1)]:
   chosen=0
   for r in candidates:
    if r['generator_family']!=f or r['visual_label']!=label or r['recording_component'] in groups:continue
    selected.append(r);groups.add(r['recording_component']);chosen+=1
    if chosen==needed:break
   assert chosen==needed
 save(a.out/'tiny_selection.json',selected);tiny=[]
 for r in selected:
  raw=payload(r);x=feature(r);y=raw['map_target'].float() if r['visual_label'] else torch.zeros(len(raw['video']),1,256,256);m=raw['map_validity'].bool()&raw['frame_validity'][:,None,None,None]
  touch(manifest.parent/r['tensor_path']);tiny.append((r,x,y.cuda(),m.cuda(),raw['video']))
 for param in model.classifier.parameters():param.requires_grad_(False)
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
 history=[];tiny_visual=[]
 def evaluate(epoch):
  model.eval();errors=[]
  with torch.no_grad():
   for r,x,y,m,video in tiny:
    pred=forward(x)['modification_map'][0].float();errors.append({'sample_id':r['sample_id'],'fake':r['visual_label'],'mae':float((pred-y).abs()[m].mean())})
    if epoch in [0,100]:
     k=int(x['frame_validity'].sum())//2;tiny_visual.append(dict(epoch=epoch,sample_id=r['sample_id'],method=r['generator_family'],video=video[k].clone(),target=y[k].cpu(),prediction=pred[k].cpu()))
  row={'epoch':epoch,'mean_clip_mae':float(np.mean([v['mae'] for v in errors])),'fake_clip_mae':float(np.mean([v['mae'] for v in errors if v['fake']])),'per_clip':errors};history.append(row);save(a.out/'tiny_history.json',history);print('tiny',epoch,row['fake_clip_mae'],flush=True)
 evaluate(0)
 for epoch in range(1,101):
  model.train();order=list(range(len(tiny)));random.shuffle(order)
  for i in order:
   r,x,y,m,video=tiny[i];opt.zero_grad(set_to_none=True);pred=forward(x)['modification_map'][0].float();loss=(pred-y).abs()[m].mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
  if epoch%5==0:evaluate(epoch);torch.save({'heads':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'config':state['config']},a.out/'tiny_last.pt')
 torch.save(tiny_visual,a.out/'tiny_visuals.pt')
 after={str(p):sha(p) for p in protected};assert before==after
 for path,stat in input_stats.items():
  s=Path(path).stat();assert (s.st_size,s.st_mtime_ns)==stat,path
 save(a.out/'complete.json',{'complete':True,'architecture_unchanged':True,'protected_checksums_match':True,'read_input_stats_unchanged':len(input_stats),'tiny_epochs':100,'tiny_clips':25,'tiny_objective':'masked map L1 only; classifier frozen; pretrained epoch-2 initialization','limitations':['Method templates use known method as a diagnostic oracle. Global templates require no method label.','Median uses middle valid frame per training fake clip; mean uses all valid frames.','Tiny-set results measure training-set fitting, not generalization.']})
if __name__=='__main__':main()
