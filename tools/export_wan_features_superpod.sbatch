#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --qos=llmsvgen_normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=dgx-10
#SBATCH --job-name=wan-feature-export
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/setup/ffpp_wan_features_fp32_v1_20261118/export-%j.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
NAME=ffpp_wan_features_fp32_v1_20261118
PY=/project/llmsvgen/junhong/venvs/wan_detector/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HOME="$ROOT/.cache/huggingface"
export PYTHONPATH="$ROOT/ours/experiments/$NAME/source"
cd "$PYTHONPATH"
"$PY" -m pytest -q tests/test_feature_cache.py
"$PY" -u "$ROOT/setup/$NAME/export_wan_features.py" \
 --run-config "$ROOT/ours/runs/ffpp_joint_wan_only_v1_20261118_786090/run_config.json" \
 --output "$ROOT/datasets/features/$NAME"
