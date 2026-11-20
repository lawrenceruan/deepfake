#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --qos=llmsvgen_normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=wan-map-diagnosis
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/setup/ffpp_other_paired_3s8fps_v1_20261118/map-diagnosis-%j.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
RUN=deepfakes_full_3s8fps_eval50_v1_20261118
export PYTHONPATH="$ROOT/ours/experiments/$RUN/source"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "$PYTHONPATH"
/project/llmsvgen/junhong/venvs/wan_detector/bin/python -u \
 "$ROOT/setup/ffpp_other_paired_3s8fps_v1_20261118/diagnose_map_prior.py" \
 --run "$ROOT/ours/runs/${RUN}_785605" \
 --output "$ROOT/ours/runs/deepfakes_map_prior_diagnostic_v1_20261118"
