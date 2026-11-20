# Dataset preservation

The user requires these rules for all dataset work in this project:

- Preserve the original raw dataset and archives unchanged.
- Start each new preprocessing task from a separately named spare copy. An
  identical resumed run may reuse its already verified spare copy.
- Read input videos without modifying them, including videos in the spare copy.
  Perform transformations in memory and save derived data to a new directory
  that is separate from both the original raw data and the spare copy.
- Use clear, versioned names for working copies, preprocessing settings,
  derived datasets, manifests and training runs.
- Preserve previous preprocessing outputs, configurations and checkpoints so
  earlier settings can be revisited. Do not overwrite them with a new setting.
- Record input checksums and verify that preprocessing leaves inputs unchanged.
