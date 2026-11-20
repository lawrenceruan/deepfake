#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=llmsvgen
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=wan-cache-audit
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/setup/ffpp_wan_features_fp32_v1_20261118/audit-%j.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
NAME=ffpp_wan_features_fp32_v1_20261118
export PYTHONPATH="$ROOT/ours/experiments/$NAME/source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
/project/llmsvgen/junhong/venvs/wan_detector/bin/python -u \
 "$ROOT/setup/$NAME/audit_wan_feature_cache.py" \
 --cache "$ROOT/datasets/features/$NAME" --experiment "$ROOT/ours/experiments/$NAME"
