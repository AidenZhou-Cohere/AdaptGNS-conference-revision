# Goop 3D source preparation and first-record timing

This family contains completed train/validation acquisition, byte-preserving
conversion, full auxiliary census, a failed first timing attempt, the corrected
timing attempt and its CPU graph-stage profile. It is preparation evidence for
a separate 3D extension, not a trained model or accuracy result.

The converted training split has 1,000 trajectories with 301 stored frames and
295,000 eligible six-frame histories. Particle counts range from 2,478 to 14,837
(median 7,436.5). The 100-trajectory validation split has particle counts from
2,588 to 13,597 (median 8,163.5). Source records, array descriptors and auxiliary
presence/bit patterns are retained in the scalar manifests and census. Context
omission follows the released parser/model contract; it is not inferred from
auxiliary constancy or NaN values.

Timing V1 failed before GPU work because the first-record inspector stored
numeric arrays under `numeric/` while its descriptors used a `train/` prefix.
V1 source, failed output and review remain preserved. Separately hashed V2
corrected only the staging-path interpretation and schema. It measured one
warmup and three updates per case on two copies of the first training record
(9,271 particles each), comparing native base against forced 25% annulus
exposure. These eight optimizer updates are infrastructure probes only.

The later CPU profile preserved input/noise, all graph-ledger fields and ordered
edge hashes. The measured mean optional-graph/ledger stage took about 0.958
seconds for base and 0.998 seconds for forced exposure; native CPU connectivity
took about 0.231 and 0.232 seconds. This identifies a bookkeeping cost on one
state. It excludes CUDA transfer, GNN, backward and Adam work and cannot be
subtracted from the GPU timing as a proven end-to-end speedup. No cohort endpoint
or full-study forecast is selected here. The subsequent optimization is outside
this snapshot.

The reviewed 3D trainer candidate in the shared source family uses 37 node and
4 edge features, radius 0.025, the D3 faithful objective, vector risk `3q`, and
separate checkpoint/RNG/schedule contracts. It requires a prospectively declared
endpoint and fresh six-model initialization; no admission or probe promotion is
provided. The frozen graph's self-edge docstring is imprecise: one self candidate
is inserted before the native cap, and sufficiently many coincident ties can
exclude it. The implementation and tests preserve that native behavior.

`raw_evidence.tar.gz` and `archive_members.json` preserve exact evidence bytes
and identities. Smaller originals are also under `records/`; array datasets and
checkpoints are excluded with hash inventories. The first-record scalar fixture
and all runnable dependencies/tests are in
[the shared source family](../cuda_graph_support_sources_20261006/README.md).
