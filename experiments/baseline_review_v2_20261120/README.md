# Baseline review — 2026-11-20

Scope: maintain the current Wan-only baseline before the proposed SAM 2.1
Hiera-Large image encoder + pretrained pyramid neck experiment. No RGB branch
or new training run was started.

## Findings and changes

| Finding | Revision | Consequence |
| --- | --- | --- |
| AUC used float32 sigmoid probabilities, which can collapse different large logits to identical 0/1 values. | Save raw logits and use them for AUC in training and evaluation. | Preserves available ranking information; old probability-only AUC cannot be exactly repaired without inference. This does not explain away classification errors or increasing BCE. |
| Direct checkpoint writes could leave a corrupt file after interruption; checkpoint and epoch report writes were separate. | Atomic same-directory replacement; commit `last.pt` with recoverable metrics, predictions and selected-checkpoint updates. | Resume can repair reports after an interrupted publication. Work within an unfinished epoch is still repeated. |
| Only minimum-loss and final weights survived, losing the peak-accuracy checkpoint seen in the dropout run. | Bounded best-accuracy and best-AUC checkpoints plus an explicit primary selection criterion. | Future experiments retain useful classification weights without saving every epoch. No historical missing weights can be recovered. |
| Split audit ignored the manifest's explicit recording-component grouping. | Audit recording components as well as source/donor/pair/identity IDs. | Rejects relatives with different video IDs when their shared component is supplied. Existing joint manifest passes. |
| Invalid zero head count could reach division by zero; list/tuple layer configuration could compare unequal at resume. | Early configuration validation and normalized layer tuples. | Clear failures and equivalent configuration comparison; existing valid architecture unchanged. |

Brier score and >=99%-confident error counts are now standard epoch metrics.
Alternative checkpoint evaluation writes into a separately named report folder.
The original zero-dropout config remains; a separate dropout-0.2 config is added.

## Validation

- 39 tests passed in the existing Superpod virtual environment, using CPU.
  Coverage includes actual Wan feature API versus block hooks on a tiny model,
  frozen parameters, cached features, padded sequences, decoder gradients,
  DSSIM, preprocessing/segments, saturation-resistant AUC and injected save
  failures. One expected warning disables CUDA autocast on CPU.
- Existing dropout `best.pt` (epoch 2) and `last.pt` (epoch 20) both loaded
  strictly into the revised heads. One existing cached validation clip produced
  finite logits and maps of shape `[1,25,1,256,256]` with both checkpoints.
- The actual combined manifest passed the stronger split audit. The cache
  loader verified its manifest and the sampled feature payload checksum.
- Both checkpoints and the combined manifest had identical hashes before and
  after these read-only checks. Core model, modules, extractor, loss, feature
  cache and preprocessing source files match the preserved pre-review hashes.
- This is CPU regression/compatibility validation, not a full CUDA retraining
  or re-evaluation of all historical predictions. Numerical equivalence to
  historical GPU BF16 outputs has not been claimed.

Evidence: `pytest.log`, `compatibility.json`, `unchanged_components.json`.
Rollback/reference: `before_revision.tar.gz`, `before_sha256.json`.
`baseline_fixes.patch` records changes relative to the backed-up core files;
the current evaluation/report tools are included in the revised source archive.
The isolated revised server source is under this experiment directory, not in
either historical training snapshot.

## Remaining design issues, not confirmed implementation bugs

1. Blurry maps: frozen Wan grids are spatially coarse and the decoder restores
   resolution by learned refinement and interpolation. L1 can favor smooth
   conditional estimates. Memorizing a tiny set supports decoder capacity but
   does not prove missing RGB is the only limitation. No attention-mask defect
   was found in the reviewed path or its padding-invariance tests.
2. Evaluation protocol: 250 entries share 50 recording groups across methods;
   reused real entries and relatives are correlated. Repeated model selection
   on this validation set cannot establish independent test generalization.
3. DSSIM measures paired-image dissimilarity, including possible residual
   alignment/compression effects. Center cropping does not guarantee perfect
   correspondence. It is not a binary ground-truth manipulation segmentation.
4. Loss reporting: map MAE is aggregated over valid pixels. With batch size 1,
   training takes per-clip normalized map losses, so short clips have different
   relative weight in optimization versus the epoch's pixel-weighted report.
   Kept unchanged to preserve the experiment definition.
5. Resume validates configuration and input paths, but does not hash every
   model/data input on each restart. It relies on the project's preservation
   rule and the cache's recorded checksums. Atomic saves protect against
   interrupted writes; they do not constitute a backup against storage loss.

Next experiment: retain this Wan-only dropout-0.2 baseline, then compare a
separate SAM 2.1 Hiera-Large + pyramid-neck RGB branch feeding the map decoder.
Predeclare the checkpoint criterion and keep splits, labels, loss and sampling
matched so the RGB branch's contribution can be assessed.
