#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=llmsvgen
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=ffpp-manifest-audit
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/setup/ffpp_other_paired_3s8fps_v1_20261118/finalize-%j.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
export PYTHONPATH="$ROOT/ours/experiments/ffpp_other_paired_3s8fps_v1_20261118/source"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
/project/llmsvgen/junhong/venvs/wan_detector/bin/python -u \
 "$ROOT/setup/ffpp_other_paired_3s8fps_v1_20261118/finalize_other_ffpp.py" --root "$ROOT"
