# Optional frozen VideoMAE-B RGB branch

Enable with `configs/wan_videomae_base_decoder_v1.json` and
`--videomae-checkpoint /path/to/videomae_base_k400_pretrained_v1_20261120`.
Install `requirements-videomae.txt` into the project environment. The official
MCG-NJU/videomae-base Kinetics-400 self-supervised checkpoint is loaded locally;
there are no implicit downloads. Only its encoder is instantiated.

`rgb_branch: none` retains the Wan-only decoder and does not import VideoMAE.
`rgb_branch: videomae_base_decoder` enables the frozen encoder in eval mode.
The RGB branch supplies only the map decoder. Classification uses Wan features;
joint map gradients can still influence shared trainable Wan adaptation layers.

Prepared RGB crops stay geometrically aligned with the DSSIM targets. The default
input is 256x256, with ImageNet mean/std from the checkpoint. Every valid frame
is used; an odd final frame is repeated to complete the last two-frame tubelet.
Invalid padded frames never enter the encoder. The checkpoint's fixed positional
grid (8x14x14) is interpolated in T/H/W to the clip's actual grid, using trilinear
interpolation with align_corners=False. This is an adaptation beyond the native
16-frame/224px checkpoint and needs empirical validation.

One unpooled final encoder grid is retained after the pretrained final LayerNorm:
`[B, ceil(valid_frames/2), 768, 16, 16]` at 256px. No RGB dataset cache is built,
in CPU RAM or on disk. For 25 frames this grid is 4.875 MiB at 16-bit precision.

Three trainable 1x1 adapters project this same compact grid to decoder channels
64/64/32 before spatial expansion. Features are interpolated to individual frame
indices using tubelet centers (.5, 2.5, ...); a repeated final frame uses its true
frame index. These skips are added at decoder resolutions /16, /8 and /4. They
are resampled versions of one feature level, not an encoder multiscale pyramid.
Wan temporal features are restored before all three stages, matching the prior
RGB decoder's temporal policy. Wan-only decoding remains unchanged.

Training uses the existing joint manifest and read-only Wan disk cache. New heads
and adapters are initialized; the stopped SAM experiment is not resumed. Raw,
spare, and prepared datasets are unchanged. Checkpoints save the trainable heads,
optimizer, epoch and RNG state plus hashes of the external pretrained weights,
configuration, preprocessing and encoder implementation. Frozen VideoMAE weights
are not duplicated inside `last.pt`. Resume after epoch five with `--resume
/path/to/last.pt --epochs 10` (the epoch argument is the new total).

Historical SAM source/configs remain for reproducibility, but its downloaded
weights and dedicated installation cache were removed at the user's request.
