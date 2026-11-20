#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --qos=llmsvgen_normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=wan-map-tests
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/ours/experiments/map_diagnostics_v1_20261119/job-%j.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT/ours/experiments/ffpp_joint_wan_only_v1_20261118/source"
/project/llmsvgen/junhong/venvs/wan_detector/bin/python -u "$ROOT/ours/experiments/map_diagnostics_v1_20261119/run_diagnostics.py" --root "$ROOT" --out "$ROOT/ours/experiments/map_diagnostics_v1_20261119/results_${SLURM_JOB_ID}"
