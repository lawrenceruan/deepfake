# SAM 2.1 RGB decoder branch, version 1

The new configuration is `configs/wan_sam21_large_decoder_v1.json`.
The original Wan-only configurations remain supported and retain their original
forward computation and head state-dict names.

## Architecture

```text
prepared RGB clip [B,T,3,256,256]
    |
    +--> frozen Wan VAE + backbone (or existing verified Wan feature cache)
    |      --> layer fusion --> spatial-temporal attention
    |                            |                  |
    |                            |                  +--> classification logit
    |                            v
    |                     temporal map features
    |                            |
    |                 restore valid frame resolution
    |                            |
    +--> frozen SAM 2.1 Hiera-Large + pretrained FPN
           framewise resize to 1024, RGB normalization
           |              |              |
         stride 16      stride 8       stride 4
           |              |              |
       1x1 adapter    1x1 adapter    1x1 adapter
           |              |              |
           +--> /16 refinement --> /8 refinement --> /4 refinement
                                                      |
                                             1-channel logits
                                                      |
                                            bilinear resize + sigmoid
                                                      |
                                         DSSIM estimate [B,T,1,256,256]
```

Only the official image encoder and FPN are instantiated. SAM's prompt encoder,
mask decoder, memory modules and tracking are not used. The fine-to-coarse FPN
outputs have 256 channels each. Per-stage adapters use 1x1 convolutions,
GroupNorm and GELU, then add their output to the corresponding Wan decoder
features before spatial refinement. Wan supplies temporal context.

The entire pretrained SAM encoder, including its neck, is frozen and remains in
evaluation mode. Spatial-temporal attention, classifier, map decoder and RGB
adapters train with the existing BCE + masked DSSIM L1 objective and dropout 0.2.
RGB features do not enter attention or the classifier. For fixed weights and
fixed Wan features, changing RGB cannot change the classification logit. During
joint training, map-loss gradients still update shared Wan adaptation, so the
trained classifier can change indirectly.

The RGB decoder restores Wan time **before** all three spatial stages so every
RGB scale is aligned to every input frame. The Wan-only decoder continues to
restore time before its final stage. This decoder scheduling difference is part
of this variant; an encoder-only causal claim would require a matched decoder
ablation. Native SAM features are resized to the original crop's /4, /8 and /16
grids within each frame chunk to reduce retained memory. The final map still
upsamples /4 logits, so fine boundary recovery is not guaranteed.

Only valid frames are encoded. Padded RGB frames never enter SAM, attention or
decoder normalization; output maps for padded frames are zero. The default
encoder frame batch size is 1. Resizing a 256px crop to SAM's 1024px input follows
the pretrained input convention; it does not create extra source-image detail.

## Dependencies and weights

Use a **new environment**. The existing baseline environment uses an older
PyTorch; do not upgrade it in place. SAM requires PyTorch >=2.5.1 and torchvision
>=0.20.1. `requirements-sam2.txt` pins official SAM source revision
`2b90b9f5ceec907a1c18123530e92e794ad901a4`.

For example, on the GPU server:

```bash
python3.10 -m venv /project/llmsvgen/junhong/venvs/wan_sam2_rgb_v1
source /project/llmsvgen/junhong/venvs/wan_sam2_rgb_v1/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
SAM2_BUILD_CUDA=0 python -m pip install -r requirements-sam2.txt -r tests/requirements-detector.txt -r tests/requirements-preprocess.txt
```

No SAM connected-component CUDA extension is needed for encoder-only use.
For live Wan extraction, also install the original Wan dependencies and a
compatible FlashAttention build. Cached Wan training does not instantiate Wan
or require FlashAttention. Run commands from the repository root.

Download the official [sam2.1_hiera_large.pt checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
to a separately named, Git-ignored checkpoint folder. Loading is strict for all
`image_encoder.*` keys; it does not silently accept random/missing encoder weights.
Frozen SAM weights are not duplicated inside every detector checkpoint. Instead,
the detector checkpoint records the pretrained file hash, official configuration
hash and encoder source hashes. Resume/inference verify that contract. Supply
`--sam2-checkpoint` to override a relocated pretrained file with identical bytes.

## Training with the existing Wan cache

Example for a **new** run; this does not resume or overwrite the old baseline:

```bash
python -m deepfake.train \
  --config configs/wan_sam21_large_decoder_v1.json \
  --checkpoint-dir /path/to/Wan2.2-TI2V-5B \
  --context /path/to/neutral_context.pt \
  --sam2-checkpoint /path/to/sam2.1_hiera_large.pt \
  --manifest /path/to/ffpp_all_paired_3s8fps_sharedholdout_v1_20261118/manifest.jsonl \
  --wan-feature-cache /path/to/ffpp_wan_features_fp32_v1_20261118 \
  --output /path/to/new_runs/wan_sam21_large_decoder_v1 \
  --epochs 20 --batch-size 1 --accumulation 4 --lr 1e-4 \
  --selection-metric loss --seed 42
```

The disk-cache loader checks extraction settings, the source manifest and text
context hashes, feature-payload checksums, prepared-input checksums, feature
layers/channels, output geometry and frame masks. It loads each feature tensor
on demand and does not retain the full ~511 GiB cache in RAM. It preserves all
cache/data files. Cached features remain paired with the unmodified RGB clip;
do not apply new image augmentations while using this fixed cache. Keep the
original Wan checkpoint directory for later live inference.

Omit `--wan-feature-cache` for live frozen Wan extraction. The old RAM-cache
option also works, but caches Wan features only. SAM features are computed
fresh from each clip; this revision does not write a SAM feature dataset.

Resume with the same settings plus `--resume /path/to/run/last.pt`. Existing
atomic checkpoint publication is retained. Wan-only checkpoint resumes remain
supported; switching a Wan-only run to RGB is a new experiment, not a resume.
Use the same cached execution path for a new Wan-only comparison to control
initialization/runtime differences against the new RGB experiment.

An optional future setting `sam2_trainable: true` is implemented, with its own
`--sam2-lr` (default 1e-5) and saved encoder/optimizer state. It is **disabled**
in the first-experiment configuration. Frame chunking alone does not bound
backward activation memory when the encoder is trainable.

## Verification

The regression tests include RGB sensitivity of maps with fixed classification,
gradient routing, frozen encoder behavior, padded-frame independence, official
checkpoint-key loading, cache compatibility, checkpoint restore and the original
Wan-only tests. To run an explicit real-weight smoke test:

```bash
PYTHONPATH=. python tools/smoke_sam2_rgb.py \
  --sam2-checkpoint /path/to/sam2.1_hiera_large.pt \
  --device cuda --frames 25 --output /path/to/new_reports/rgb_smoke.json
```

This uses synthetic RGB and Wan grids to check forward/backward behavior; it
does not measure detection accuracy or prove improvement in map quality. The
report output must be new. No dataset preprocessing or full training is started
by installing or testing this branch.

Official references: [configuration](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2.1/sam2.1_hiera_l.yaml),
[image encoder and FPN](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/backbones/image_encoder.py),
[input transforms](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/utils/transforms.py).
