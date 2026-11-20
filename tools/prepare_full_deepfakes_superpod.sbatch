#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=llmsvgen
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=wan-full-preprocess
#SBATCH --output=/project/llmsvgen/junhong/Video_Deepfake/setup/deepfakes_full_3s8fps_eval50_v1_20261118/preprocess-%j.log
set -euo pipefail
ROOT=/project/llmsvgen/junhong/Video_Deepfake
RUN=deepfakes_full_3s8fps_eval50_v1_20261118
EXP="$ROOT/ours/experiments/$RUN"
RAW="$ROOT/datasets/raw/ffpp_c23_kaggle_20261118_v1"
PY=/project/llmsvgen/junhong/venvs/wan_detector/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$EXP/source"
"$PY" -m pytest tests -q
# Only this new, isolated raw copy is extracted. Existing raw folders are untouched.
if ! test -f "$RAW/extraction_complete.json"; then
 "$PY" - "$RAW" "$EXP/expected_raw_sha256.json" <<'PY'
import json,sys,zipfile,shutil
from pathlib import Path
from deepfake.preprocess import fingerprint
root=Path(sys.argv[1]);expected=json.loads(Path(sys.argv[2]).read_text())
with zipfile.ZipFile(root/'ff-c23_version1.zip') as archive:
 for rel in expected:
  target=root/rel
  target.parent.mkdir(parents=True,exist_ok=True)
  with archive.open('FaceForensics++_C23/'+rel) as source, target.open('wb') as dest:
   shutil.copyfileobj(source,dest)
assert all(fingerprint(root/rel)==sig for rel,sig in expected.items())
(root/'extraction_complete.json').write_text(json.dumps(dict(files=len(expected),sha256_verified=True)))
print('Raw copy: all 2000 checksums verified',flush=True)
PY
fi
"$PY" -u -m deepfake.prepare_full_segments --raw-root "$RAW" \
 --pairs "$EXP/candidate_pairs.jsonl" --expected-raw-sha256 "$EXP/expected_raw_sha256.json" \
 --output "$ROOT/datasets/preprocessed/$RUN" --config configs/preprocess_3s.json \
 --workers 8 --eval-clips 50 --seed 42 "$@"
