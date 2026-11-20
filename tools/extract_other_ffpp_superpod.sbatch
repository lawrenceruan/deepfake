#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=llmsvgen
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=ffpp-other-extract
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/setup/ffpp_other_paired_3s8fps_v1_20261118/extract-%j.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
EXP="$ROOT/ours/experiments/ffpp_other_paired_3s8fps_v1_20261118"
PY=/project/llmsvgen/junhong/venvs/wan_detector/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
cd "$EXP/source"
"$PY" -m pytest tests -q
"$PY" -u -m tools.extract_other_ffpp_raw \
 --archive "$ROOT/datasets/raw/ffpp_c23_kaggle_20261118_v1/ff-c23_version1.zip" \
 --expected "$EXP/expected_raw_sha256.json" \
 --output "$ROOT/datasets/raw/ffpp_c23_other_paired_20261118_v1"
