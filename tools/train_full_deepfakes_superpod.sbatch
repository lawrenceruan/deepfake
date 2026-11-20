#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --qos=llmsvgen_normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --job-name=wan-full-deepfakes
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/setup/deepfakes_full_3s8fps_eval50_v1_20261118/train-%j.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
RUN=deepfakes_full_3s8fps_eval50_v1_20261118
PY=/project/llmsvgen/junhong/venvs/wan_detector/bin/python
DATA="$ROOT/datasets/preprocessed/$RUN"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HOME="$ROOT/.cache/huggingface"
cd "$ROOT/ours/experiments/$RUN/source"
"$PY" - "$DATA" <<'PY'
import json,sys,torch
from pathlib import Path
from deepfake.data import audit_splits
root=Path(sys.argv[1]);s=json.loads((root/'summary.json').read_text())
assert s['complete'] and s['candidate_pairs']==1000 and s['active_clips']['val']==50
assert s['raw_sha256_verified_before_and_after'] and s['loader_verified']
audit_splits(root/'manifest.jsonl')
# Wan residual-stream hidden features are FP32 despite BF16 attention/linears.
cache_bytes=sum(s['active_clips'].values())*3*3072*7*8*8*4
assert cache_bytes<210*2**30, 'Feature cache too large for this job allocation'
print('GPU',torch.cuda.get_device_name(),'feature cache upper bound GiB',cache_bytes/2**30,flush=True)
PY
"$PY" -u -m deepfake.train --config configs/wan_only.json \
 --checkpoint-dir "$ROOT/checkpoints/Wan2.2-TI2V-5B" \
 --context "$ROOT/setup/wan_detector/neutral_context.pt" \
 --manifest "$DATA/manifest.jsonl" --preprocessing-config "$DATA/preprocessing_config.json" \
 --output "$ROOT/ours/runs/${RUN}_${SLURM_JOB_ID}" \
 --epochs 20 --batch-size 1 --accumulation 4 --lr 0.0001 --seed 42 --cache-features
"$PY" -u -m deepfake.evaluate_full_run --run "$ROOT/ours/runs/${RUN}_${SLURM_JOB_ID}"
