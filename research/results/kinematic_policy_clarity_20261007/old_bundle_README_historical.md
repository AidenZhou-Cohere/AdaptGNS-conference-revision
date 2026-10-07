# NON-ANONYMOUS author-review bundle: source and compact full evidence

This is **not a submission archive**. It preserves author names, public repository links, original local provenance paths, and all included license/attribution notices. It does not certify anonymity, redistribution rights, venue eligibility, conference readiness or the paper's claims.

The bundle contains 175 explicitly allowed tracked files from commit `d9d3a5155a02f455fb52c5b241a2e203e7afe137`. It includes the current simulator/research source, tests, immutable protocols, dependency pins and notices, together with both complete new `full_evaluation_20261005` and `full_evaluation_analysis_20261005` subtrees. Original source/evidence bytes are preserved, including eight failed rollout outcomes and the initial failed audit attempt. Existing repository READMEs retain their wider repository scope; use this file to understand the archive.

All older tracked numeric/derived evidence and model assets are omitted uniformly and identified by commit/path/size/SHA256 in `SOURCE_PROVENANCE.json`. No favorable-outcome selection is applied. The bundle omits raw/converted datasets, checkpoint weights, full per-step/frame JSON and NPZ output files, process logs, environments, Git internals and private authoring documents. The complete omitted full-evaluation raw-output inventory is included with the compact results. Some earlier-paper tables depend on the separately preserved older evidence; this archive does not reproduce those tables. The included source retains the original protocols for a later full rerun.

The new compact evidence suffices to replay the public manuscript table/prose renderer exactly. It does not suffice to rerun the strict full raw-record/array audits or trained-model inference: those require the inventoried raw outputs, official data and final checkpoints (or retraining). The official test source was accessed only after six 100k training completions. Prior inspection of indices0–2 and historical aggregate results rules out a pristine independent-confirmation claim for indices3–29.

## Verify after extraction

Use Python3.12 with the recorded dependencies in `research/requirements-local-lock.txt`; this is the observed local environment, not a portable CUDA recipe. Tests use synthetic CPU arrays and models, never trained checkpoints or benchmark data. From the extracted root, with an existing suitable environment:

```sh
export PYTHONPATH="$PWD:$PWD/adaptive-gns"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=0
export CUDA_VISIBLE_DEVICES=""
python bundle_tools/build_author_review_bundle.py verify --tree .
python research/results/full_evaluation_20261005/verify_compact_publication.py
python -m pytest -p no:cacheprovider -q research/tests/test_budget_graph.py research/tests/test_full_rollout.py research/tests/test_full_same_state.py research/tests/test_summarize_full_rollouts.py research/tests/test_summarize_full_same_state.py research/tests/test_evaluation_queue_parsers.py
python -m research.render_full_evaluation_tables --output-dir /tmp/full-evaluation-tables
```

The build checks Python syntax, import resolution inside the extraction, CLI parsers, relevant synthetic tests, the original compact publication manifest, and exact table-generation hashes against the included `paper_results_generation.json`. Generated inserts are included in `generated_tables/`. No LaTeX compilation, model inference, training, full raw-record scientific audit or network action is performed by the builder. `BUNDLE_VALIDATION.json` records what passed without nondeterministic elapsed times.

`MANIFEST.json` hashes every payload file except itself. ZIP entries are sorted with fixed timestamps and modes. The original compact package's own manifest/audit remain byte-identical. Absolute local paths in original evidence identify provenance and are not portable paths. Dataset distribution terms remain unverified; preserved software licenses are not assumed to license datasets. The existing notices are retained without rights certification.

## Rebuild the exact allowed commit

```sh
python bundle_tools/build_author_review_bundle.py build --repo /path/to/checkout --commit d9d3a5155a02f455fb52c5b241a2e203e7afe137 --allowlist bundle_tools/author_review_allowlist.json --work-dir /path/to/new-empty-build-directory --python /path/to/checked-python
```

The builder requires an explicit full commit, HEAD equality, source bytes identical to Git, complete current new-evidence subtrees, no untracked allowlist entries, no symlinks, and every tracked license/notice. A later commit needs a reviewed refreshed allowlist. The work directory must be new and outside the checkout, so previous failures and packages are preserved.
