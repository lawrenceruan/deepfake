"""Read-only monitor; records snapshots and reports new epochs or state changes."""
import json, subprocess, time
from pathlib import Path
out=Path(__file__).resolve().parent
remote=r'''
import json,subprocess,ast
from pathlib import Path
from datetime import datetime
root=Path('/project/llmsvgen/junhong/Video_Deepfake');run=root/'ours/runs/ffpp_dropout02_v1_20261119_789314'
state=subprocess.check_output(['sacct','-n','-X','-j','789314','-P','--format=JobID,State,Elapsed,ExitCode'],text=True).strip()
metrics=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()] if (run/'metrics.jsonl').exists() else []
log=root/'ours/experiments/ffpp_dropout02_v1_20261119/train-789314.log'
with log.open('rb') as f:
 f.seek(max(0,log.stat().st_size-12000));lines=f.read().decode(errors='replace').splitlines()
step=None
for line in reversed(lines):
 if line.startswith("{'epoch':"):
  try:v=ast.literal_eval(line)
  except (ValueError,SyntaxError):continue
  if 'step' in v:step=v;break
print(json.dumps(dict(checked_at_server=datetime.now().isoformat(),state=state,metrics=metrics,latest_step=step,comparison_ready=(run/'dropout_comparison.json').exists(),tail=lines[-3:])))
'''
ssh=['ssh','-o','HostName=143.89.184.2','-o','HostKeyAlias=superpod.ust.hk','-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=1','superpod.ust.hk','source /etc/profile; module load slurm; python -']
previous=None;failures=0
while True:
 try:
  p=subprocess.run(ssh,input=remote,text=True,capture_output=True,timeout=35)
  if p.returncode:raise RuntimeError(p.stderr.strip())
  r=json.loads(p.stdout);failures=0
  (out/'latest_progress.json').write_text(json.dumps(r,indent=2)+'\n')
  signature=(r['state'].split('|')[1],len(r['metrics']),r['comparison_ready'])
  if signature!=previous:
   print(json.dumps(r),flush=True);previous=signature
  if signature[0] not in ['RUNNING','PENDING','COMPLETING','REQUEUED','CONFIGURING']:
   break
 except (subprocess.TimeoutExpired,RuntimeError,ValueError,IndexError) as e:
  failures+=1;print(json.dumps({'monitor_error':str(e),'consecutive_failures':failures}),flush=True)
  if failures>=5:break
 time.sleep(45)
