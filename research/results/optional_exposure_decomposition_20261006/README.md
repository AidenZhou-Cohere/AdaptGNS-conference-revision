# Exploratory optional-exposure decomposition

This post-inspection study partitions paired errors from the six original 100k models' saved, no-loop, observed-history predictions. It performs no new model inference. All six models, all 27 test trajectories (indices 3–29), all 11 fixed target frames, all three comparisons and all seven group records are retained. The test histories and previous action-benefit results were inspected before this protocol; these results are not independent confirmation.

The CPU saved-array analysis took 42.0156 s; the separate independent audit took 34.1073 s. These operational timings are separate from model inference latency.

`report.md` is the exact original scientific report, with every sign and reported outcome unchanged. `results.json` preserves the original top-level values, all objective summaries and all model aggregates except repeated per-trajectory records and frame indices. `results.json.gz` decompresses to the exact original 12,870,423-byte result (SHA256 `71c8924bd17f1cd85a69736611e66ea442bc265f12cb920ebcfb7ab1fd508c65`), including all trajectory aggregates, frame indices, null coverage, checkpoint/configuration identities, sample SDs and all three seed values.

## Design and interpretation

The contrasts are risk minus random, risk minus speed, and speed minus random. Risk is the original `previous-observed-base-risk25` policy. Positive error differences favor the right policy. Error, alignment and perturbation cost use normalized acceleration coordinate units, dividing the previous vector quantities by two. Error difference equals cost difference minus alignment difference.

The four primary groups (`neither`, `left_only`, `right_only`, `both`) partition particles by direct optional-edge incidence. Each contribution is sum(mask × paired quantity)/N; the four contributions sum to the whole-frame difference. `both_less`, `both_equal`, and `both_more` partition only the `both` group. The primary and supplementary groups must not be added together. A particle without a directly incident optional edge can still be affected through message passing.

Frames are averaged equally within each trajectory, trajectories equally within each seed, and the three seed means are summarized by mean and sample SD. The histories and particles are repeated observations, not independent replications. Empty groups have zero fractions/contributions and null conditional means. Strict conditional means require every frame in the fixed population; separately named weighted conditionals divide the weighted contribution by the weighted group fraction. No available-frame averaging, particle pooling, significance claim or multiplicity-adjusted inference is introduced.

This is a descriptive partition by policy-defined exposure. It does not adjust for pretreatment confounding, identify marginal edge value or a causal concentration mechanism, establish autonomous stability, or measure inference speed. It does not replace the original fixed experiment, its eight NLL rollout failures, the separate native self-loop study, or the 110k continuation study.

## Exact evidence and omission scope

- `frame_records.tar.gz` contains all **1,782 exact original per-frame JSON files**, with all three comparisons, primary and supplementary group outcomes, input/derived-array hashes and **136 null metric entries**. `frame_archive_index.json` records each archive member's size, hash and null count. Empty-group null outcomes are retained.
- `status.json`, `report.md`, `input_identity.json`, `all_inputs_sha256.json`, `independent_audit.json`, `independent_audit.py`, `independent_audit_tests.py`, `pre_execution_review.json`, `protocol.md`, analysis source and tests preserve exact bytes. Every existing `attempt_*.json` in the audit directory is copied to `audit_attempts/`, including unsuccessful audit attempts if present. `provenance.json` links the source freeze `0f3589f` and separate audit.
- `raw_inventory.json` lists **all 3,569 raw files** and 304,551,831 bytes, with SHA256 and explicit included/omitted locations. Derived particle NPZ arrays remain unchanged locally; every NPZ identity is linked from its archived frame record. Other nonembedded outputs, if present, are also inventoried. This package retains all scalar/group/null outcomes without duplicating particle vectors.
- No checkpoints, datasets or previous raw same-state/action-benefit inputs are embedded. Their preserved hashes identify required assets but are not download links. Full particle arithmetic reauditing requires those external assets and omitted arrays. The curation verifies stable output bytes; it does not rerun the separate full-array audit or rehash dataset/checkpoint payloads.

Run `python3 verify_publication.py` in this directory for the standard-library package check. It verifies every packaged hash, exact decompressed result/projection, original status/audit linkage, all 1,782 exact JSON records, null accounting and omitted-array identities without extracting the archive or loading models. `PUBLICATION_MANIFEST.json` and `PUBLICATION_AUDIT.json` record package identity and the raw byte-stability check.

## Reproduction

The frozen implementation is `research/analyze_optional_exposure_decomposition.py`; `protocol.md` gives the complete contract. With the original saved inputs available, run from the repository root:

```sh
python -m research.analyze_optional_exposure_decomposition \
  --evaluation-root /path/to/original/full-same-state \
  --action-root /path/to/original/action-benefit \
  --output-dir /path/to/new-exposure-output
```

The command requires a fresh output directory, all original fixed-population inputs and pinned hashes. Absolute paths in exact originals are provenance strings and may require the recorded layout for exact identity replay. The separate audit script accepts explicit analysis, repository and output paths plus expected source/protocol hashes; it also requires the original saved inputs. This compact evidence package is not a standalone rerun bundle. Research-fork artifacts are nonanonymous; author verification, anonymity and asset rights remain submission checks.
