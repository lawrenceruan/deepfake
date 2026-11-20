#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --qos=llmsvgen_normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=wan-deepfakes-trial
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/setup/wan_detector/train-%j.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
PY=/project/llmsvgen/junhong/venvs/wan_detector/bin/python
DATA="$ROOT/datasets/preprocessed/deepfakes_trial_100train_20eval_v1"
CKPT="$ROOT/checkpoints/Wan2.2-TI2V-5B"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/.cache/huggingface"
export WANDB_MODE=disabled
cd "$ROOT/ours"
RESUME_ARGS=()
if [[ $# -ge 1 ]]; then
    RESUME_ARGS=(--resume "$1")
fi
EPOCHS="${2:-5}"
"$PY" - "$DATA" <<'PY'
import json,sys,torch
from pathlib import Path
from deepfake.data import audit_splits
root=Path(sys.argv[1])
summary=json.loads((root/'summary.json').read_text())
assert summary['complete'] and summary['accepted_pairs']=={'train':100,'val':20}
audit_splits(root/'manifest.jsonl')
assert torch.cuda.is_available()
print('GPU:',torch.cuda.get_device_name(),flush=True)
PY
if ! test -f "$ROOT/setup/wan_detector/neutral_context.pt"; then
    "$PY" -m deepfake.cache_context --checkpoint-dir "$CKPT" \
        --output "$ROOT/setup/wan_detector/neutral_context.pt"
fi
"$PY" -u -m deepfake.train --config configs/wan_only.json \
    --checkpoint-dir "$CKPT" --context "$ROOT/setup/wan_detector/neutral_context.pt" \
    --manifest "$DATA/manifest.jsonl" --output "$ROOT/ours/runs/deepfakes_trial_${SLURM_JOB_ID}" \
    --epochs "$EPOCHS" --batch-size 1 --accumulation 4 --lr 0.0001 --seed 42 --cache-features \
    "${RESUME_ARGS[@]}"
