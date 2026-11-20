#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=llmsvgen
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --job-name=ffpp-other-prepare
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/setup/ffpp_other_paired_3s8fps_v1_20261118/preprocess-%A_%a.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
EXP="$ROOT/ours/experiments/ffpp_other_paired_3s8fps_v1_20261118"
PY=/project/llmsvgen/junhong/venvs/wan_detector/bin/python
FAMILIES=(Face2Face FaceSwap FaceShifter NeuralTextures)
FAMILY=${FAMILIES[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
cd "$EXP/source"
"$PY" -u -m deepfake.prepare_full_segments \
 --raw-root "$ROOT/datasets/raw/ffpp_c23_other_paired_20261118_v1" \
 --pairs "$EXP/${FAMILY}_pairs.jsonl" --expected-raw-sha256 "$EXP/${FAMILY}_expected_raw_sha256.json" \
 --output "$ROOT/datasets/preprocessed/${FAMILY,,}_full_3s8fps_sharedholdout_v1_20261118" \
 --config configs/preprocess_3s.json --workers 8 --eval-clips 50 --seed 42 \
 --split-reference "$ROOT/datasets/preprocessed/deepfakes_full_3s8fps_eval50_v1_20261118/manifest.jsonl" "$@"
