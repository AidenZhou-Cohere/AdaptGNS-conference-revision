# Data inputs

The distribution contains scientific summaries, not raw benchmark data or model weights. Obtain benchmark files from their original publishers and retain their terms: WaterDrop and Goop are from the Learning to Simulate Complex Physics dataset; the Sand data used here are the DesignSafe GNS dataset family (project PRJ-3702). The manuscript cites the dataset and simulator sources. No download happens when running the saved-result commands.

Simulation entry points consume a version1 `gns-trajectory-manifest` JSON file with its matching adjacent `metadata.json` and safe numeric `.npy` arrays. Each split declares `format`, `version`, `dataset`, `split`, `source`, `record_count`, `metadata`, `metadata_sha256`, and ordered `records`. Each record preserves its original zero-based `source_index`, split ID such as `train:000000`, position/type shape and dtype, relative array path, byte length, SHA256, and combined trajectory-content SHA256. Goop also retains its unused `step_context` array and its hash; that auxiliary field is checked and excluded from the model.

The combined content hash is SHA256 of the ASCII string `positions_sha256:particle_types_sha256`. Paths resolve within the manifest directory. Position arrays are float32 `[T,N,D]`; Goop uses full int64 type vectors. Sand's original scalar or vector int32/int64 type representation is retained. The loader verifies every array and metadata hash before simulation; metadata headers alone do not establish that the arrays are valid.

Training uses all 1000 ordered Goop or Sand trajectories. The complete 2D validation/test source has30 trajectories; six observed frames are followed by395 Goop,314 Sand or995 WaterDrop forecast frames. Goop3D validation and test each contain100 source records; the fixed 30-source grid is listed separately in `../protocols/goop3d_inputs.json`. Never replace that grid by the first30 records.

WaterDrop conversion uses the CRC32C-checked reader with complete source files named `train.tfrecord`, `valid.tfrecord` and `test.tfrecord` in the input directory:

```sh
PYTHONPATH=code:code/adaptive-gns python -m research.prepare_full_waterdrop \
  --input-dir /downloads/WaterDrop --output-dir /data/WaterDrop \
  --metadata /downloads/WaterDrop/metadata.json --splits train valid test
```

The converter checks record CRCs, source completion, duplicate trajectories, shapes and finite positions, and records source/array SHA256s. Compare the resulting source identity with the intended official download before training or evaluation. The same saved numerical source routines define the trajectory schema. For other materials, preserve the publisher's source order and all auxiliary fields when converting; conversion is a separate input-preparation step, not part of the lightweight saved-results command. The portable training command expects those complete converted manifests. Model checkpoints must contain their saved architecture, normalization, optimizer/update identity and the relevant training configuration; a bare state dictionary is insufficient for the study checks.

Dataset/model availability and permission are separate from code portability. This package grants no additional dataset or checkpoint redistribution rights.
