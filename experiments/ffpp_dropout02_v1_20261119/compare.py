import sys,json
from pathlib import Path
r=Path(sys.argv[1]);baseline=r.parents[1]/'runs/ffpp_joint_wan_only_v1_20261118_786090'
def summarize(p):
 metrics=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()];results=[]
 for m in metrics:
  pred=json.loads((p/f"val_predictions_epoch_{m['epoch']:03d}.json").read_text());m=dict(m)
  m['brier']=sum((x['fake_probability']-x['label'])**2 for x in pred)/len(pred)
  m['confident_errors_99']=sum((x['fake_probability']>=.5)!=bool(x['label']) and max(x['fake_probability'],1-x['fake_probability'])>=.99 for x in pred);results.append(m)
 return {'best_by_loss':min(results,key=lambda m:m['val_loss']),'last':results[-1],'all_epochs':results}
out={'baseline_dropout_0':summarize(baseline),'dropout_02':summarize(r),'note':'Same 250 validation entries, 50 recording groups; threshold 0.5; validation selects best checkpoint.'}
(r/'dropout_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:{s:v[s] for s in ['best_by_loss','last']} for k,v in out.items() if isinstance(v,dict)}),flush=True)
