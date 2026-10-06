# Sand graph-support capacity and failed timing gate

This is the faithful base/mix graph-support preparation family, separate from
the earlier native-base faithful/NLL capacity study. It preserves numerical
implementation checks, six 512-update capacity probes, synthetic terminal-ledger
I/O timing, and complete evaluation-timing evidence including failures.

All six graph-support capacity probes verified. Their training-only planning
term is 26,412.736 seconds under `1.35 * (100000*q6 + 12*r6)`, with
`q6=0.19001664200914092` and `r6=46.94376225024462`. The terminal-ledger experiment
measured synthetic serialization/I/O overhead; it did not train or validate a
100k scientific model.

The V1 complete-horizon evaluation timing gate **failed**. Of 108 required
314-step rollouts, 63 completed, 39 failed a coordinate resource guard and
6 failed a candidate-pair resource guard. All 90 same-state and 90 clean
validation cases completed. Shorter accepted prefixes do not establish the
full-horizon runtime or accuracy. The final reviewed full-horizon runtime
forecast and complete-study fit remain null. Later quota-v2 preparation is a
separate allocation policy with no complete-outcome promise; it does not repair
or reinterpret this failed V1 gate.

The archive includes final raw timing rows/snapshots/logs, all adverse outcomes,
protocols/releases, capacity histories/configurations and independent reviews.
Earlier incomplete snapshot files are preserved when their bytes differ; when
identical, `omitted_inventory.json` points to the exact retained final copy.
Historical draft records retain their original wording. Checkpoints and data
arrays are excluded; hashes, descriptors and local omitted-file identities are
retained. No SSH/connection/authentication files are distributed.

`raw_evidence.tar.gz` is lossless, and `archive_members.json` records each
original preparation-relative member name, SHA256 and byte count. `records/`
contains exact copies of compact reports for review. Verification and the
unchanged runnable code/tests are in
[the shared source family](../cuda_graph_support_sources_20261006/README.md).
Earlier strict CPU/CUDA numerical failures and the different native-base
capacity study remain in the sibling `sand_cuda_validation_20261006`,
`sand_cuda_numerical_mechanism_20261006` and `sand_cuda_capacity_20261006` families.
