#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --qos=llmsvgen_normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=24:00:00
#SBATCH --job-name=wan-dropout02
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/ours/experiments/ffpp_dropout02_v1_20261119/train-%j.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
EXP="$ROOT/ours/experiments/ffpp_dropout02_v1_20261119"
PY=/project/llmsvgen/junhong/venvs/wan_detector/bin/python
OUT="${1:-$ROOT/ours/runs/ffpp_dropout02_v1_20261119_$SLURM_JOB_ID}"
DATA="$ROOT/datasets/manifests/ffpp_all_paired_3s8fps_sharedholdout_v1_20261118"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled HF_HOME="$ROOT/.cache/huggingface"
export PYTHONPATH="$EXP/source"
export MPLCONFIGDIR="$EXP/matplotlib"
cd "$EXP/source"
"$PY" "$EXP/preflight.py" "$EXP"
RESUME=()
if [ -f "$OUT/last.pt" ]; then RESUME=(--resume "$OUT/last.pt"); fi
"$PY" -u "$EXP/train_cached.py" --config "$EXP/dropout02.json" --checkpoint-dir "$ROOT/checkpoints/Wan2.2-TI2V-5B" --context "$ROOT/setup/wan_detector/neutral_context.pt" --manifest "$DATA/manifest.jsonl" --preprocessing-config "$DATA/preprocessing_config.json" --output "$OUT" --epochs 20 --batch-size 1 --accumulation 4 --lr 0.0001 --seed 42 --cache-features "${RESUME[@]}"
"$PY" -u "$ROOT/setup/ffpp_joint_wan_only_v1_20261118/evaluate_joint_ffpp.py" --run "$OUT"
"$PY" -u "$ROOT/setup/ffpp_joint_wan_only_v1_20261118/report_joint_ffpp.py" --run "$OUT"
"$PY" -u "$ROOT/setup/ffpp_joint_wan_only_v1_20261118/plot_joint_ffpp.py" --run "$OUT"
"$PY" -u "$EXP/compare.py" "$OUT"
