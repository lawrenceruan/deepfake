#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --account=llmsvgen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=wan-detector-check
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/setup/wan_detector/smoke-%j.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
PY=/project/llmsvgen/junhong/venvs/wan_detector/bin/python
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
cd "$ROOT/ours"
if ! test -f "$ROOT/setup/wan_detector/neutral_context.pt"; then
    "$PY" -m deepfake.cache_context --checkpoint-dir "$ROOT/checkpoints/Wan2.2-TI2V-5B" \
        --output "$ROOT/setup/wan_detector/neutral_context.pt"
fi
"$PY" -u - <<'PY'
import json,time,torch
from pathlib import Path
from deepfake.config import DetectorConfig
from deepfake.backbone import FrozenWanFeatures
from deepfake.model import WanOnlyDetector
from deepfake.losses import detector_loss
root=Path('/project/llmsvgen/junhong/Video_Deepfake')
cfg=DetectorConfig.load('configs/wan_only.json')
t=time.time()
extractor=FrozenWanFeatures.from_pretrained(root/'checkpoints/Wan2.2-TI2V-5B',root/'setup/wan_detector/neutral_context.pt',cfg)
model=WanOnlyDetector(extractor,cfg).train()
model.heads.cuda()
p=sorted((root/'datasets/preprocessed/deepfakes_trial_100train_20eval_v1/clips').glob('*paired_fake.pt'))[0]
payload=torch.load(p,weights_only=True,map_location='cuda')
with torch.autocast('cuda',dtype=torch.bfloat16):
    result=model(payload['video'][None],payload['frame_validity'][None])
    losses=detector_loss(result,torch.ones(1,device='cuda'),payload['map_target'][None],
                        payload['map_validity'][None],payload['frame_validity'][None])
losses['loss'].backward()
assert torch.isfinite(losses['loss'])
assert all(p.grad is None for p in model.extractor.parameters())
assert model.heads.fusion.fuse.weight.grad.abs().sum()>0
assert all(torch.isfinite(p.grad).all() for p in model.heads.parameters() if p.grad is not None)
report=dict(status='PASS',gpu=torch.cuda.get_device_name(),seconds=time.time()-t,
    peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,map_shape=list(result['modification_map'].shape),
    loss=float(losses['loss']),input=str(p),trained_detector=False,
    note='Pretrained frozen Wan forward and random detector-head backward; no optimizer step or trained checkpoint.')
(root/'setup/wan_detector/gpu_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
PY
