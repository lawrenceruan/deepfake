"""Read-only compact progress stream for one submitted experiment."""
import argparse,json,time,subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--name',required=True)
p.add_argument('--prep',required=True)
p.add_argument('--train',required=True)
a=p.parse_args()
setup=a.root/'setup'/a.name
run=a.root/'ours/runs'/f'{a.name}_{a.train}'
previous=None
for _ in range(24*60):
    status=subprocess.run(['squeue','-h','-j',f'{a.prep},{a.train}','-o','%i %T %M %R'],capture_output=True,text=True).stdout.strip()
    progress={}
    prep_log=setup/f'preprocess-{a.prep}.log'
    if prep_log.exists():
        for line in prep_log.read_text(errors='replace').splitlines()[-20:]:
            if line.startswith('{"completed":'):
                progress['preprocessing']=json.loads(line)
    train_log=setup/f'train-{a.train}.log'
    if train_log.exists():
        import ast
        for line in train_log.read_text(errors='replace').splitlines()[-15:]:
            if line.startswith("{'epoch':"):
                try:
                    item=ast.literal_eval(line)
                    progress['training']={k:item[k] for k in ('epoch','step','loss','classification','map') if k in item}
                except (ValueError,SyntaxError):pass
    metrics=run/'metrics.jsonl'
    if metrics.exists() and metrics.stat().st_size:
        item=json.loads(metrics.read_text().splitlines()[-1])
        progress['latest_epoch']={k:item[k] for k in ('epoch','train_loss','val_loss','accuracy','auroc','fake_map_mae')}
    record=dict(queue=status,**progress)
    print(json.dumps(record),flush=True)
    if not status:
        finished=subprocess.run(['sacct','-n','-j',f'{a.prep},{a.train}','--format=JobID,State,ExitCode,Elapsed'],capture_output=True,text=True).stdout
        print(finished,flush=True)
        for log in (prep_log,train_log):
            if log.exists():print(log.name+'\n'+'\n'.join(log.read_text(errors='replace').splitlines()[-5:]),flush=True)
        break
    time.sleep(55)
