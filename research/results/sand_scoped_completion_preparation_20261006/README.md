# Sand completion tools

This source-only package completes the previously prepared paired Sand study:
six fresh base/mix models at 100,000 updates, three seeds and all six fixed
evaluation policies. It contains no new experimental outcomes. Training and
evaluation require separate releases tied to the actual completed models,
available devices and remaining time.

The four current entries are in
`workspace/work/deadline_research_20261005/cuda_preparation/`:

| Entry | Purpose |
| --- | --- |
| `prepare_sand_final_cohort_scoped_v1.py` | Verify and combine the completed six-model cohort. |
| `prepare_sand_reserved_test_scoped_v1.py` | Acquire, convert and admit the reserved test source after cohort completion. |
| `supervise_sand_final_evaluation_scoped_v1.py` | Run the original complete comparison under explicit GPU ownership and fixed quotas. |
| `summarize_sand_graph_support_scoped_v1.py` | Collect stopped artifacts and summarize all paired results, failures, physical diagnostics and measured costs. |

The operational sequence and incomplete release templates are beside those
entries. `current_approved_pins.json` identifies the reviewed versions.
Independent review records and unsuccessful preparation attempts are retained;
they are software evidence, not scientific findings. One large raw test log
remains in the original workspace and is identified by size and hash in
`large_local_log_references.json`.

Restore the exact predecessor workspace files in `dependency_paths.json` into a
new workspace, then overlay this package's `workspace/`. The dependencies are
published in the neighboring `goop3d_vectorized_preparation_20261006` and
`scoped_execution_completion_preparation_20261006` families. Do not overwrite
active experiment sources. Keep the lexical virtual-environment Python path
when executing reviewed operations; resolving its symlink can select system
Python instead.

Run `python3 verify_bundle.py` from this directory to verify package and
dependency bytes. `packaging_checks.json` describes the bounded checks performed
on a reconstructed source workspace. The scientific sources and protocols are
unchanged. Preparation checks do not establish dataset admission, numerical
accuracy, complete evaluation, an isolated runtime measurement or submission
readiness.

The fixed schedule places Sand seed 0 on aquamarine GPUs 2/3 after the original
Goop training supervisor has fully exited, and seeds 1/2 on yellow after its
Goop GPU evaluation. All six models and policies remain required. Shared-host
timings must retain that scope. No two-GPU-only fallback is implied.

Full datasets, model checkpoints, raw numerical outputs, credentials, SSH
configuration and private process inventories are excluded. Referenced assets
must be supplied separately for scientific reproduction.
