#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --qos=llmsvgen_normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --job-name=wan-25frame-smoke
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/setup/deepfakes_full_3s8fps_eval50_v1_20261118/smoke-%j.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
RUN=deepfakes_full_3s8fps_eval50_v1_20261118
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$ROOT/ours/experiments/$RUN/source"
/project/llmsvgen/junhong/venvs/wan_detector/bin/python -u - "$ROOT" "$RUN" <<'PY'
import json,sys,torch
from pathlib import Path
from deepfake.backbone import FrozenWanFeatures
from deepfake.config import DetectorConfig
from deepfake.model import WanOnlyDetector
from deepfake.losses import detector_loss
root=Path(sys.argv[1]);run=sys.argv[2]
clip=torch.load(root/'setup'/run/'smoke_25frames_fake.pt',weights_only=True,map_location='cpu')
video=(clip['video'].float()/255).unsqueeze(0).cuda()
target=clip['map_target'].unsqueeze(0).cuda()
mask=clip['map_validity'].unsqueeze(0).cuda()
cfg=DetectorConfig.load('configs/wan_only.json')
model=WanOnlyDetector(FrozenWanFeatures.from_pretrained(root/'checkpoints/Wan2.2-TI2V-5B',root/'setup/wan_detector/neutral_context.pt',cfg),cfg)
model.heads.cuda();model.train()
records=[]
for length in (25,10):
 valid=(torch.arange(25,device='cuda')[None]<length)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  grids,latent_valid=model.extractor(video,valid)
  output=model.heads(grids,latent_valid,(25,256,256),valid)
  loss=detector_loss(output,torch.ones(1,device='cuda'),target,mask,valid)
 assert output['modification_map'].shape==(1,25,1,256,256)
 assert torch.isfinite(loss['loss'])
 loss['loss'].backward()
 assert all(p.grad is None for p in model.extractor.parameters())
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.heads.parameters())
 records.append(dict(valid_frames=length,feature_shapes=[list(g.shape) for g in grids],
    feature_dtypes=[str(g.dtype) for g in grids],loss=float(loss['loss']),valid_map_pixels=int(loss['valid_map_pixels'])))
 model.zero_grad(set_to_none=True)
result=dict(gpu=torch.cuda.get_device_name(),passed=True,cases=records,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
(root/'setup'/run/'gpu_smoke_verified.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
PY
