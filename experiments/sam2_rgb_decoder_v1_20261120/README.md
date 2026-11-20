# SAM 2.1 Hiera-Large decoder branch: implementation validation

Configuration: `configs/wan_sam21_large_decoder_v1.json`. SAM encoder and neck
are frozen; dropout is 0.2. RGB supplies three feature scales to the map decoder.
The classifier consumes Wan features only. See `docs/SAM2_RGB.md` for details.

Validation on 2026-11-20:

- 47 unit/regression tests passed across the original baseline and new RGB/cache
  tests; a separate actual-trainer integration test also passed (48 total).
- The integration test compares two uninterrupted epochs with one epoch followed
  by resume from `last.pt`. Tiny CPU encoder fixtures are used, while dataset
  loading, optimization, dropout RNG, checkpoint publication and restore are
  real. Final head weights and metrics match exactly. CUDA execution itself is
  not exercised by this test.
- The official pretrained SAM 2.1 Hiera-Large encoder plus FPN contains
  **212,703,280 frozen parameters**; the default trainable detector has
  **5,538,034 parameters**.
- Real-weight CPU smoke tests at SAM input size 1024 passed for 1 and 25 frames.
  The 25-frame test produced `[1,25,1,256,256]` maps, finite losses and nonzero
  gradients in all three RGB adapters. SAM parameters received no gradients.
  This uses synthetic RGB and Wan grids, not a dataset evaluation.
- The test environment is isolated from the existing training environment:
  `/private/tmp/deepfake-sam2-test-20261120`, Python 3.13 and PyTorch 2.15.0.
  SAM source is pinned to `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- The checkpoint SHA256 is
  `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`.
  The downloaded checkpoint is in a Git-ignored weights folder.

Local, Git-ignored evidence: `pytest.log`, `resume_test.log`,
`pretrained_cpu_smoke_v1.json`, `pretrained_cpu_25frames_v1.json`, and
`baseline_equivalence.json`. The latter checks initialized head parameters and
valid/padded outputs against the published baseline commit with RGB disabled.

No raw/derived dataset, existing feature cache or historical checkpoint was
modified. No full training or GPU performance evaluation was started. Map quality
improvement and classification changes remain questions for the next experiment.
