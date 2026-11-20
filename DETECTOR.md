# Wan-only detector: setting A

This document describes the preserved Wan-only baseline. For the optional
SAM 2.1 Hiera-Large RGB decoder branch, see [SAM2_RGB.md](docs/SAM2_RGB.md).

This is the first detector configuration discussed for the project. No RGB
encoder, audio branch, LoRA, reconstruction objective, or real/fake feature
distribution regularization is enabled.

## Implemented path

```
face clip [B,T,3,H,W], RGB in [0,1]
  -> frozen Wan2.2 VAE (official normalization/scaling)
  -> fixed clean-latent probe (optional fixed flow sigma)
  -> frozen Wan2.2-TI2V-5B blocks 9,19,29, before output head
  -> unpadded grids [B,3072,T',H/32,W/32]
  -> per-depth LayerNorm + Linear(3072,256)
  -> concatenate + Linear(768,256)
  -> axis-specific positional encoding
  -> 2 x [spatial attention -> temporal attention -> FFN]
       all pre-norm + residual, 8 heads
       -> spatial/valid-temporal mean pooling -> classifier logit [B]
       -> progressive decoder -> sigmoid map [B,T,1,H,W]
```

Both objectives update fusion and attention. Only the downstream modules are
trainable. Frozen modules remain in eval mode when the detector enters train
mode. Wan's standard generation forward is preserved; `feature_layers` enables
an early feature-only return. Package exports are lazy so model-only imports do
not eagerly import speech/animation pipelines.

The decoder uses a low-resolution 3D context convolution and spatial refinement
at /16, /8 and /4. It restores original frame count before /4 refinement and
upsamples one-channel logits to full resolution. Temporal interpolation uses
the causal VAE grid's first-frame/4-frame-group convention, with anchors at
0,4,8,... . This is a proposed decoder approximation, not exact inversion of
temporal compression. Maps describe estimated modification strength, not
calibrated forgery probability.

## Requirements and checkpoints

Run full extraction on a CUDA machine (the intended H800). Use upstream
`requirements.txt`, including FlashAttention, in a separate Python 3.10+ GPU
environment. CUDA/FlashAttention compatibility must match that machine.
Download the **official-format** `Wan-AI/Wan2.2-TI2V-5B` checkpoint, not the
separately converted Diffusers-format checkpoint. The checkpoint directory
must contain the DiT config/weights, `Wan2.2_VAE.pth`, and T5/tokenizer files.

The local implementation/tests do not download the 5B weights or start training.
An empty neutral text prompt is encoded once with the official T5 encoder:

```bash
cd /path/to/Video_Deepfake/ours
python -m deepfake.cache_context \
  --checkpoint-dir /path/to/Wan2.2-TI2V-5B \
  --output data/neutral_context.pt
```

This separate process releases T5 before loading the detector. Cached context
must come from the same checkpoint and be reused in training and inference.
The main DiT runs under BF16 autocast. Its time embedding/projection remain
FP32, as required by upstream forward. Loss reductions are FP32.

`probe_sigma=0` is the initial clean probe. For nonzero sigma we use
`z_sigma=(1-sigma)*z+sigma*epsilon`, `t=1000*sigma`, matching the upstream flow
scheduler's post-shift sigma convention. This is not an index into a denoising
schedule. A fixed seed produces the same noise for every same-shaped sample,
independent of batch position. No denoising sequence or VAE decoding is run.

## Prepared data contract

### Configured three-second segmentation

The new entry point `python -m deepfake.segment_preprocess` loads
`configs/preprocess_3s.json` by default. `--config` selects another contract;
optional `--duration`, `--sample-fps`, and `--size` override its values. Defaults
are three seconds, 8 FPS, 256x256 RGB in [0,1]. Maximum frames are **derived**
as `duration * sampling_fps + 1` (25), validated against Wan's `4n+1` constraint.
`PreprocessingConfig` is dataset-independent; dataset adapters are responsible
for pairing and alignment. The implemented raw-video entry point currently
supports FF++ paired videos, not arbitrary external dataset layouts.

This command decodes timestamps, plans all segments, selects matched source
frames, applies shared spatial crops/resizing, and computes per-frame DSSIM for
each segment. No intermediate MP4 files or video re-encoding are needed.
Short videos and tails retain their speed: repeat-last padding fills 25 tensor
positions, with padded frames/maps invalid and padded targets zero. Sampling
uses inclusive endpoints; adjacent full windows share one boundary sample.
Do not create a tail consisting only of an already sampled endpoint. Low-FPS
sources can reuse frames during resampling; this is logged separately from
trailing padding. Decode timestamp failures are rejected, not silently replaced
by assumed FPS. FF++ pairing still uses a screened same-index common prefix;
general temporal remapping is not implemented.

Changing the configuration does **not** retroactively change saved clips. The
old trial tensors contain only 17 sampled frames, so regenerating three-second
segments requires reading the raw recordings again. To preserve the exact
recording selection and train/evaluation assignments, pass the old manifest:

```bash
python -m deepfake.segment_preprocess \
  --config configs/preprocess_3s.json \
  --raw-root ../datasets/ff-c23/FaceForensics++_C23 \
  --pairs ../datasets/preprocessed/deepfakes_trial_100train_20eval_v1/manifest.jsonl \
  --output ../datasets/preprocessed/deepfakes_segments_3s_v1
```

Old clip rows collapse to unique raw recording pairs; real companion rows are
skipped. All new segments inherit the pair's split. A new output directory is
required; raw data and old prepared versions remain untouched. Rejections may
reduce usable recording coverage; inspect reports before training. The above
full regeneration command is documented, not automatically launched.

Each run stores the resolved `preprocessing_config.json` and its SHA256
fingerprint in every manifest row. The loader rejects mixed fingerprints within
a split and accepts `expected_preprocessing_id=cfg.fingerprint` to enforce a
specific contract. Legacy manifests remain loadable without that optional check.
The old `deepfake.preprocess` and `deepfake.prepare_trial` commands retain their
explicit frame-index behavior for reproducibility; use `segment_preprocess`
for the new configured workflow. Existing training jobs/data are not migrated.

The training loader consumes **already cropped/aligned face clips**, not raw
Kaggle MP4s. `deepfake.preprocess` prepares selected paired clips in a separate
stage, with reference-only face detection and automatic correspondence screening.
Do not silently use independently cropped frames for paired targets.

### Agreed spatial preprocessing policy

Preserve the center-crop correction for FF++ pairs whose fake frame has the
same height and a narrower width than its real reference. The audit found
evidence of this geometry in sampled Face2Face and NeuralTextures pairs.
For a verified pair, crop the real reference to the fake width with left offset
`(real_width - fake_width) // 2`; for example, 720x480 to 704x480 removes
8 pixels from each side. Do not squeeze the full real frame to the fake width.
Equal-size pairs need no dimension correction, but still require alignment checks.
Other size mismatches require a separately verified correspondence rule.

The preprocessing order is: establish frame correspondence, validate and apply
the center crop where applicable, apply shared face-track crop coordinates to
both images, resize both face crops identically to the model input resolution,
then compute DSSIM. Record the applied geometry and verify alignment before
setting `alignment_verified: true`. Matching dimensions alone are insufficient;
unverified pairs must not supply dense DSSIM supervision.

### Raw-video preprocessing

Install `tests/requirements-preprocess.txt` alongside the detector dependencies.
From `ours`, for an explicit candidate JSONL selection:

```bash
python -m deepfake.preprocess \
  --raw-root ../datasets/ff-c23/FaceForensics++_C23 \
  --pairs ../datasets/preprocessed/smoke_pairs.jsonl \
  --output ../datasets/preprocessed/ffpp_dssim_smoke_v2 \
  --frames 17 --start-frame 32 --frame-stride 2 --size 256 --split smoke
```

Each candidate needs `video_path` and `reference_path`, relative to the raw root.
The output must be a **new directory disjoint from the raw root**. The tool never
rewrites raw MP4s. It writes `clips/*.pt`, `manifest.jsonl`, fixed-scale DSSIM
preview PNGs, per-pair geometry/timestamp/quality reports, source SHA256 hashes,
configuration, and a summary. It checks source hashes before/after successful
decoding and loads every accepted tensor through `PreparedClipDataset`.

The current implementation processes one requested clip per candidate row:

1. Decode corresponding frame indices sequentially, without independent FPS
   resampling. Out-of-range clips are rejected; extra unmatched tails are unused.
2. Center-crop the reference horizontally to the fake width when heights match.
   Reject other geometry. Do not infer correctness from dimensions alone.
3. Detect faces on reference frames using OpenCV's bundled frontal-face Haar
   cascade. Build one fixed square tube enclosing the detections with 1.4x
   margin, and use it for both videos. Reject missing/ambiguous detections,
   large tracking jumps, or tubes that require padding.
4. Check RGB background correspondence outside an expanded face tube at 128x96.
   Defaults: mean error <=0.04, 95th-percentile error <=0.12, background coverage
   >=15%, and background texture standard deviation >=0.02. Reject clearly
   better alternative reference frames within +/-2 indices. These thresholds
   are screening defaults, not calibrated alignment guarantees. Near ties are
   logged as timing ambiguity; static scenes cannot prove temporal alignment.
5. Apply the same tube to real and fake, resize both using area interpolation,
   and compute continuous DSSIM at the final resolution. Mask the outer five
   pixels because the 11x11 SSIM window touches the artificial crop boundary.
6. Save a fake clip with its cached target and its corresponding real clip with
   a zero target. Their shared pair/source identifiers keep them together in
   split checks. No labels/maps are generated for rejected clips.

`alignment_verified` here means the documented automatic checks passed, **not
manual certification**. Review previews and reports before scaling up. The Haar
detector is a lightweight initial implementation and will reject profile,
occluded, or difficult faces; a stronger tracker is future work. Local offset
checks do not search for global temporal remapping. The default `smoke` split
is solely for pipeline testing; official splits and donor/identity leakage
auditing must be supplied before training. Multiple manipulations of one source
produce corresponding real crops per pair; deduplication/sampling policy remains
a training-dataset decision. Neither Wan weights nor a GPU are needed here.

Smoke validation (2026-11-17): nine candidate pairs tested across two runs in
`../datasets/preprocessed/`; five accepted pairs produced ten 17-frame 256x256
clips. Accepted: Deepfakes 000_003 and 017_803, FaceShifter 000_003, Face2Face
021_312, NeuralTextures 021_312. The latter two use a 654-to-640 center crop
(seven pixels per side). Face2Face/NeuralTextures 005_010 and 007_132 were
rejected by face detection. All ten clips loaded/batched, cached maps matched
recomputation exactly, accepted input SHA256 hashes remained unchanged, and
first/middle/last previews were inspected. Thirteen CPU tests passed including
five preprocessing tests. Background timing near-ties remain reported in the
accepted examples; this run does not certify exact timing or full-video alignment.

### First single-method training trial

`deepfake.prepare_trial` creates the Deepfakes-only pilot with 100 accepted
training pairs and 20 accepted evaluation pairs (200/40 clips, including real
references). It reserves 20% of target/donor connected components before
preprocessing, uses seed 42, and skips rejected candidates within their assigned
pool. Unused held-out components never enter training. The loader audits donor
recording IDs as well as target recording IDs. This prevents recording leakage;
it does not establish person-identity separation when identity labels are absent.

The prepared trial is `../datasets/preprocessed/deepfakes_trial_100train_20eval_v1`.
It required 166 candidates, with 46 rejected. It has 100 training target recordings
and 20 evaluation target recordings; the 190/38 participating target/donor
recordings are disjoint. Sampling remains 17 frames, indices 32 through 64 at
stride 2, with 256x256 crops. This is a small clip-level pilot, not evaluation
over entire videos or the official FF++ protocol.

`tools/train_deepfakes_superpod.sbatch` defaults to five epochs on one Slurm GPU using
the standalone virtual environment `/project/llmsvgen/junhong/venvs/wan_detector`.
The project root is `/project/llmsvgen/junhong/Video_Deepfake`. It uses AdamW,
learning rate 1e-4, batch size 1, accumulation 4, and seed 42. `--cache-features`
keeps frozen Wan outputs in CPU RAM after their first extraction; it supports
fixed unaugmented batch-one clips, and the cache is rebuilt after a restart.
Classification accuracy/AUROC/confusion counts, global and fake-only map MAE,
the zero-map fake MAE baseline, and per-clip validation predictions are saved
each epoch. The 20-pair `val` set is monitored for checkpoint selection, so it
is a validation set, not an untouched final test set. Actual GPU results should
be read from the run's `metrics.jsonl`; preparation alone is not trained-model
validation.

The account's authorized `normal` partition and `llmsvgen_normal_qos` were used
after the preempt queue reported a long wait. Job 785575 completed five epochs
on H800 in 2m03s, then job 785587 resumed the same optimizer/model state to 20
total epochs in another 4m00s. Both exited 0:0. Resume usage is
`sbatch tools/train_deepfakes_superpod.sbatch /absolute/path/to/last.pt 20`.

Best checkpoint by combined validation loss: epoch 19 (one-based). Independently
verified accuracy 40/40 (20 real, 20 fake) and AUROC 1.00. Global map MAE was
0.017645; fake-only map MAE 0.027684 versus 0.043199 for an all-zero fake-map
prediction. The first five epochs had only 50% threshold accuracy, so the short
startup run was insufficient. Peak allocated GPU memory was about 12.7 GiB.
Seventeen CPU tests passed both locally and in the standalone remote environment,
and the real pretrained Wan GPU forward/backward check passed. All 240 transferred
tensors matched their SHA256 checksums; source/donor splits were disjoint and a
separate exact-RGB-tensor check found no cross-split duplicates.

Remote full checkpoints and predictions are in
`/project/llmsvgen/junhong/Video_Deepfake/ours/runs/deepfakes_trial_785587/`.
The smaller `detector_best.pt` export is for inference; full `best.pt` and `last.pt`
retain optimizer/RNG state for resuming. These results concern one fixed clip per
video and a small monitored validation set from one manipulation method. They do
not establish official-benchmark performance, identity separation, or transfer
to other manipulation types.

Standalone inference job 785592 also completed (0:0) after reloading the best
checkpoint. On held-out pair 012_026, fake scores were 0.999527 for the fake and
0.002397 for the real; they matched saved validation predictions within 1e-6.
The inspected map captures broad eye/mouth regions but is substantially smoother
than its DSSIM target. Local `runs/deepfakes_trial_785587/` contains the inference
export, prediction records, `trial_report.json`, and `heldout_map_preview.png`.

Use a JSONL manifest with one line per clip, for example:

```json
{"sample_id":"001_fake_clip0","tensor_path":"clips/001_fake_clip0.pt","split":"train","visual_label":1,"sample_type":"paired_fake","source_video_id":"001","pair_id":"001_clip0","identity_ids":["person_a","person_b"],"alignment_verified":true}
```

Tensor files are dictionaries saved with `torch.save`:

- `video`: float `[T,3,H,W]` in `[0,1]`; H,W divisible by 32.
- `reference`: optional identically sampled/cropped real RGB tensor.
- `map_target`: optional precomputed `[T,1,H,W]` map, instead of reference.
- `map_validity`: optional boolean `[T,1,H,W]`; mark uncertain correspondence,
  crop borders and other unusable pixels false.
- `frame_validity`: optional boolean `[T]`; only right-padding supported.

Supported rows:

| sample_type | visual_label | Supervision |
|---|---:|---|
| real | 0 | BCE plus genuine zero-map target on valid pixels |
| paired_fake | 1 | BCE plus DSSIM; requires alignment_verified |
| unpaired_fake | 1 | BCE only, regardless of any placeholder map |

Generated DSSIM targets use channel-mean `(1-SSIM)/2`, an 11x11 Gaussian
window (sigma 1.5), RGB data range 1, constants 0.01² and 0.03², reflect borders,
population local moments, and fixed [0,1] scaling. No per-image normalization
or binarization is applied. This differs by a factor of two from DiffusionFF's
`1-SSIM` convention. Cached targets require manifest field
`map_convention: "rgb_mean_half_dssim_gaussian11_sigma1.5"`.

C23-derived targets include compression effects. Validity masks are supplied
by preprocessing; the loader cannot infer trustworthy alignment from labels.

The loader pads batches and masks invalid frames. The frozen extractor probes
each sample independently so another sample's padding cannot affect its Wan
attention. Non-4n+1 clip lengths repeat the last valid frame only to the next
4n+1 boundary. Output/loss padding is excluded. This handles short clips at the
cost of one backbone call per batch member; use batch 1 + accumulation first.

Training audits source/pair/known identity overlap across splits. Populate
identity_ids with **both** donor and target identities if known. Missing identity
metadata cannot be audited automatically. Real/fake variants must stay together.

## Train and infer

```bash
python -m deepfake.train \
  --config configs/wan_only.json \
  --checkpoint-dir /path/to/Wan2.2-TI2V-5B \
  --context data/neutral_context.pt \
  --manifest data/clips.jsonl \
  --output runs/wan_only --batch-size 1 --accumulation 4

# Resume at an epoch boundary:
# add --resume runs/wan_only/last.pt

python -m deepfake.infer \
  --checkpoint-dir /path/to/Wan2.2-TI2V-5B \
  --context data/neutral_context.pt \
  --detector runs/wan_only/best.pt \
  --clip data/clips/suspect.pt --output runs/suspect_prediction.pt
```

Training uses AdamW, gradient accumulation/clipping, BCE plus masked L1, and
validation loss for best-checkpoint selection. It logs separate real/fake map
losses and peak GPU memory. Checkpoints contain downstream weights, optimizer,
config and RNG states; they do not duplicate frozen 5B weights. Resume is
supported at epoch boundaries, not mid-batch. No learning-rate scheduler is
used. Inference returns an uncalibrated sigmoid score and maps for one clip.
Long-video window aggregation and original-frame overlay export remain future
integration work; no end-to-end raw-video inference is claimed.

## Verification

```bash
python -m pytest tests/test_detector.py tests/test_wan_feature_api.py -q
```

The CPU suite checks tensor shapes, classification/map gradients reaching shared
modules, frozen weights, padding invariance, DSSIM identity/locality, missing
supervision, split leakage, small synthetic-subset overfit and weight reload.
It also runs the actual WanModel forward with tiny random weights and a CPU
SDPA replacement for FlashAttention, compares feature outputs to block hooks,
and checks ordinary generation output remains unchanged.

These tests do not establish trained accuracy, full-checkpoint compatibility,
GPU memory fit, or CUDA numerical behavior. Before a full run, perform a real
checkpoint forward/backward memory pilot at the configured frame count and
resolution, then an overfit check on a small verified FF++ subset.

Local verification (2026-11-17): 8 tests passed on CPU with PyTorch 2.15.0,
Diffusers 0.43.0 and pytest 8.4.2 in an isolated temporary environment. CUDA
autocast warnings in the tiny Wan test are expected on this Mac. The upstream
batched timestep broadcast was also corrected from `[B]` to `[B,1]` before
expanding over tokens.

## Full Deepfakes run: three-second segments

The `deepfakes_full_3s8fps_eval50_v1_20261118` experiment preserves the older
17-frame trial and processes all 1,000 Deepfakes candidate recording pairs.
`configs/preprocess_3s.json` specifies 3-second windows at 8 FPS, including
both endpoints (25 frames). Short tails retain their actual timing; repeated
trailing frames are invalid padding. Reference center cropping, the shared
face tube, correspondence screening and the DSSIM definition remain the same.
Quality-screening rejections are recorded separately from accepted segments.

`deepfake.prepare_full_segments` processes recording pairs in parallel, saves
a completion record per pair for resuming, and checks every raw SHA-256 before
and after processing. Use a new output directory for every changed setting.
`--resume` is only for the same configuration and output directory. On Superpod,
use eight workers in the CPU queue and export `OPENCV_FFMPEG_THREADS=1` to avoid
each worker spawning another 16 decoder threads.

New tensors store RGB losslessly as uint8 with the explicit
`video_encoding="uint8_rgb_0_255"` tag. The loader restores the exact float32
uint8/255 values used previously. Fake DSSIM targets remain float32; real
zero-map targets may be omitted and are reconstructed by the loader. Older
float RGB tensors remain supported.

The split selects **50 evaluation clips total**, balanced 25 real/25 fake,
with one clip from each of 50 target/donor connected recording components.
It prefers full windows and uses seed 42, without using model scores or map
values. All other clips from those components are saved in
`reserved_holdout_relatives.jsonl` and excluded from training. The remaining
accepted clips form training. This enforces recording separation; unknown
person identities are not certified as disjoint.

`tools/prepare_full_deepfakes_superpod.sbatch` and
`tools/train_full_deepfakes_superpod.sbatch` contain the versioned server paths.
The detector starts with fresh heads, uses the existing frozen Wan checkpoint,
and trains for 20 epochs with batch 1, accumulation 4 and learning rate 1e-4.
Fixed frozen features are cached in CPU RAM. Training requires the prepared
data's preprocessing fingerprint through `--preprocessing-config`. A fresh
run refuses a nonempty output directory. Training aggregates every batch into
full-epoch metrics: classification weighted by clip count and map error
weighted by valid-pixel count. These are online training measurements while
weights change; evaluation uses the final weights of each epoch.

The 50 clips are monitored validation data used to select the checkpoint.
After training, `deepfake.evaluate_full_run` reloads the selected checkpoint,
verifies all 50 saved predictions and exports examples covering map-error
quantiles and classification errors. `tools/plot_full_run.py` creates the
loss curves and fixed-scale DSSIM comparison figures. Experiment notes,
source snapshots, raw checksums and preservation checks are saved separately
under `experiments/deepfakes_full_3s8fps_eval50_v1_20261118/`.

Verification for this setting: 26 CPU tests passed locally and in the Junhong
virtual environment. H800 job 785598 passed full 25-frame and masked 10-frame
forward/backward checks at 256x256; frozen modules had no gradients, and peak
allocated GPU memory was 12.67 GiB. The full-clip features were three FP32 grids
of shape `[1,3072,7,8,8]`.

## Additional paired FF++ methods and shared holdout

The `ffpp_other_paired_3s8fps_v1_20261118` experiment processes Face2Face,
FaceSwap, FaceShifter and NeuralTextures separately, with the same preprocessing
fingerprint as the full Deepfakes run. Raw extraction goes into a new
`ffpp_c23_other_paired_20261118_v1` directory and is checked against 5,000 local
SHA-256 hashes (four fake families plus originals). The existing archive and
Deepfakes outputs are read-only inputs. DeepFakeDetection remains pending:
the Kaggle mirror does not include its paired actor-original videos.

Use `--split-reference` with the original Deepfakes manifest when preparing
another method. This fixes the same 50 held-out target/donor components and
evaluation labels across methods. Selection prefers the same parent and
segment, then another valid clip of the required label within that component.
If a component has no accepted clip, its evaluation item is reported missing;
it is never replaced by a training component. All relatives remain excluded
from training. A combined multi-method evaluation contains repeated recording
groups across methods, not 200 independent recording groups.

The extraction and four-job array are in `tools/extract_other_ffpp_superpod.sbatch`
and `tools/prepare_other_ffpp_superpod.sbatch`. Versioned outputs are named
`{method}_full_3s8fps_sharedholdout_v1_20261118`. The 27-test CPU suite includes
a missing-group cross-method leakage check. Local first-segment smoke checks
passed for all four methods, including loader verification.

## Diagnosing map priors without retraining

`tools/diagnose_map_prior.py` reads a completed run, leaves its checkpoint and
dataset unchanged, and writes to a new output folder. It fits two fixed spatial
templates using only training fake targets: a mean over all valid frames and
a median over one middle valid frame per training clip. All held-out model
predictions are rechecked against the original run before comparing fake-only
masked MAE. A middle-frame wrong-clip control swaps predictions among fake
clips without using evaluation targets to fit a template. Correlation and
temporal-variation statistics provide additional descriptions.

These controls test whether outputs depend on the clip. They do not establish
artifact-specific segmentation: an input-conditioned anatomical prior can
also adapt to face position, size and pose. DSSIM is paired-image dissimilarity,
not a binary manipulation annotation. Treat the monitored 50 clips as validation
data, since they selected the original checkpoint.

## Joint training of all five paired methods

The separate experiment `ffpp_joint_wan_only_v1_20261118` trains fresh detector
heads on Deepfakes, Face2Face, FaceSwap, FaceShifter and NeuralTextures together.
It uses an exact copy of the previous validated training source and the same
20 epochs, AdamW learning rate 1e-4, batch size 1, accumulation 4, seed 42,
frozen Wan VAE/backbone and BCE + masked DSSIM L1 objective. No RGB encoder is
added. Inputs come from the audited combined metadata-only manifest.

All manifest entries are uniformly shuffled; per-method paired real entries
are retained, including possible repeated original crops. The five validation
sets each contain 50 entries and share the same 50 target/donor recording
components. Consequently the 250 validation entries are not 250 independent
recordings. All relatives of those components remain excluded from training.
The checkpoint is selected using aggregate validation loss, and metrics are
also reported separately by method.

`tools/train_joint_ffpp_superpod.sbatch` is gated on the successful final data
audit. `evaluate_joint_ffpp.py` rechecks all checkpoint predictions and exports
per-method map-error quantiles, real controls and classification errors.
`report_joint_ffpp.py` reports per-method metrics for every epoch;
`plot_joint_ffpp.py` renders loss curves and a separate example figure for each
method. Every artifact is saved under the new run; previous data and model
runs stay unchanged. DeepFakeDetection remains excluded pending paired actor
originals.

## Reusable frozen Wan feature cache

The completed cache is on Superpod at
`/project/llmsvgen/junhong/Video_Deepfake/datasets/features/ffpp_wan_features_fp32_v1_20261118`.
It contains all 36,004 active entries (35,754 train and 250 validation) from the
joint FF++ manifest. Reserved holdout relatives are excluded. The three Wan
layer grids (9, 19, 29) retain FP32 values, with latent/frame validity masks,
clip IDs, output dimensions and links to the unchanged prepared targets.
Feature files total 511.45 GiB.

`deepfake.feature_cache.CachedWanFeatureDataset(root, split, extraction_id)`
loads verified features after checking the completion marker and manifest
checksum. It returns unbatched grids; see the experiment's `usage.txt` for a
batch-one detector-head example. Cache metadata records model/context hashes,
preprocessing, probing, source-code and precision settings. Changing detector
heads or losses can reuse the cache; changing the Wan inputs, weights, probe
settings or requesting uncached layers requires a compatible new extraction.

Export job 786148 ran independently on dgx-25 and completed in 2:53:48. Audit
job 786292 passed. The original training job 786090 on dgx-10 was not modified
or interrupted. All entries passed exact serialization checks; the final audit
verified the full metadata/split catalog and 38 representative input/payload
checksums, including full and short clips. Reports and the local metadata copy
are under `experiments/ffpp_wan_features_fp32_v1_20261118`.

## Baseline maintenance revision (2026-11-20)

The review and preserved source backup are in
`experiments/baseline_review_v2_20261120`. The architecture, frozen extractor,
loss, preprocessing and existing feature values are unchanged. The optional
`configs/wan_only_dropout02_v2.json` makes dropout 0.2 explicit; the original
zero-dropout configuration is preserved.

New training reports retain raw classification logits and use them for AUC,
avoiding ranking ties introduced by saturated sigmoid probabilities. Accuracy
still uses probability >= 0.5. Brier score and the number of wrong predictions
with confidence >= 99% are also recorded. Historical probability-only reports
remain supported and are not retrospectively corrected.

`last.pt` is written by atomic replacement and includes the completed epoch's
metrics and predictions. Resume can replay report/checkpoint publication if
interruption occurs after `last.pt` commits. An interruption before that commit
resumes from the preceding completed epoch, not from the interrupted minibatch.
`best.pt` defaults to lowest validation loss. `--selection-metric accuracy` or
`--selection-metric auroc` chooses another criterion for a new run. The trainer
also retains `best_accuracy.pt` and `best_auroc.pt`; this is a bounded set, not
an archive of every epoch. Legacy resumes can retain these additional best
weights only from the point of upgrade; previously discarded weights cannot
be recovered. Keep the selection metric fixed when resuming.

Both evaluation entry points accept `--checkpoint` and `--output`. Evaluating
a non-default checkpoint requires a fresh report directory, preventing it
from replacing the default checkpoint's report. Exported head filenames identify
the chosen checkpoint. The joint reporting script continues to aggregate the
original run directory's epoch records.

The split audit now checks `recording_component` in addition to source, donor,
pair and known identity IDs. The existing joint manifest passes this stronger
check. Validation remains shared across 50 recording groups and is used for
model selection; it is not an independent final test set.
