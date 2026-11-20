"""Read-only monitoring of the independent export and existing training."""
import json
from pathlib import Path
import subprocess
import time

r = Path('/project/llmsvgen/junhong/Video_Deepfake')
name = 'ffpp_wan_features_fp32_v1_20261118'
out = r/'datasets/features'/name
for _ in range(530):
    result = subprocess.run(['squeue', '-h', '-j', '786148,786090', '-o', '%i %T %M %R'],
                            capture_output=True, text=True)
    if result.returncode:
        print(json.dumps(dict(monitor_error=result.stderr)), flush=True)
        time.sleep(55)
        continue
    q = result.stdout.strip()
    record = dict(queue=q)
    for file in ('progress.json', 'complete.json'):
        p = out/file
        if p.exists():
            record[file] = json.loads(p.read_text())
    p = r/'setup/ffpp_joint_wan_only_v1_20261118/train-786090.log'
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0, p.stat().st_size-2000))
            record['training_latest'] = f.read().decode(errors='replace').splitlines()[-1]
    print(json.dumps(record), flush=True)
    if not any(line.split()[0] == '786148' for line in q.splitlines() if line.strip()):
        print(subprocess.run(['sacct', '-j', '786148', '--format=JobID,State,ExitCode,Elapsed,MaxRSS', '-P'],
                             capture_output=True, text=True).stdout, flush=True)
        p = r/'setup'/name/'export-786148.log'
        if p.exists():
            print('\n'.join(p.read_text(errors='replace').splitlines()[-12:]), flush=True)
        break
    time.sleep(55)
