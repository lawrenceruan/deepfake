#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --qos=llmsvgen_normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=24:00:00
#SBATCH --job-name=wan-joint-ffpp
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/setup/ffpp_joint_wan_only_v1_20261118/train-%j.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
RUN=ffpp_joint_wan_only_v1_20261118
PY=/project/llmsvgen/junhong/venvs/wan_detector/bin/python
DATA="$ROOT/datasets/manifests/ffpp_all_paired_3s8fps_sharedholdout_v1_20261118"
OUT="$ROOT/ours/runs/${RUN}_${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HOME="$ROOT/.cache/huggingface"
cd "$ROOT/ours/experiments/$RUN/source"
export PYTHONPATH="$PWD"
"$PY" - "$DATA" <<'PY'
import json,sys,torch
from pathlib import Path
from deepfake.data import audit_splits
root=Path(sys.argv[1]);s=json.loads((root/'summary.json').read_text())
assert s['complete'] and s['metadata_only']
assert set(s['families'])=={'Deepfakes','Face2Face','FaceSwap','FaceShifter','NeuralTextures'}
assert s['heldout_recording_groups']==50 and not s['shared_train_heldout_recordings']
assert s['active_clips']['val']==250
assert len(s['loader_checked_samples'])==20
audit_splits(root/'manifest.jsonl')
cache_bytes=sum(s['active_clips'].values())*3*3072*7*8*8*4
assert cache_bytes<700*2**30, 'Feature cache exceeds allocated memory budget'
print(json.dumps(dict(gpu=torch.cuda.get_device_name(),active_clips=s['active_clips'],
    feature_cache_upper_bound_gib=cache_bytes/2**30,heldout_recording_groups=50)),flush=True)
PY
"$PY" -u -m deepfake.train --config configs/wan_only.json \
 --checkpoint-dir "$ROOT/checkpoints/Wan2.2-TI2V-5B" \
 --context "$ROOT/setup/wan_detector/neutral_context.pt" \
 --manifest "$DATA/manifest.jsonl" --preprocessing-config "$DATA/preprocessing_config.json" \
 --output "$OUT" --epochs 20 --batch-size 1 --accumulation 4 --lr 0.0001 --seed 42 --cache-features
"$PY" -u "$ROOT/setup/$RUN/evaluate_joint_ffpp.py" --run "$OUT"
"$PY" -u "$ROOT/setup/$RUN/report_joint_ffpp.py" --run "$OUT"
export MPLCONFIGDIR="$ROOT/setup/$RUN/matplotlib"
"$PY" -u "$ROOT/setup/$RUN/plot_joint_ffpp.py" --run "$OUT"
