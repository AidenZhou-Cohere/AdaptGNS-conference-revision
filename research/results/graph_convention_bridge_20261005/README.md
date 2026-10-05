# Exploratory graph-convention bridge

Restoring native self-messages lowers observed-history base prediction error, but previous-observed-risk allocation still has higher mean error than random allocation in every objective/seed on both splits. The six original final 100k checkpoints are unchanged. This follow-up was designed after inspection of the completed fixed study; it does not replace or repair its original results or eight NLL rollout failures.

Each model evaluates the same 128 validation histories (30 trajectories) and 297 existing test histories (27 trajectories, indices 3–29) under all 16 declared graph cases. **2,550 frames and 40,800 case outcomes completed, with zero failed/missing cases; all 2,550 native identity/parity gates passed.** These are paired repeated observations from six models, not 40,800 independent statistical samples. `coverage.json` provides every model/split/case count, including zeros.

The table reports decoder-equivalent normalized-acceleration coordinate MSE, with equal-history averages within each trajectory, equal-trajectory averages within each seed, then mean ± sample SD across seeds 0–2. Positive risk-minus-random differences favor random. Base reduction is computed as 100×(loop-off−loop-on)/loop-off **inside each seed**, followed by the seed mean and sample SD; it is not a ratio of group means. All16 cases, all seed/trajectory summaries, 37 paired contrasts, signs, null/coverage accounting and graph diagnostics are retained in `results.json`; the exact original per-frame summary is retained in `graph_convention_bridge.json.gz`.

| Split | Objective | Base, loops off | Base, loops on | Within-seed base reduction | Previous risk minus random, loops on |
|---|---|---:|---:|---:|---:|
| valid | faithful | 0.010553652 ± 0.00043402866 | 0.0086427241 ± 0.00047950937 | 18.131 ± 1.9193% | 0.00087758238 ± 0.00020271618 |
| valid | nll | 0.011425601 ± 0.00031625571 | 0.0094865454 ± 0.00052811387 | 17.012 ± 2.4348% | 0.0015811174 ± 0.00052462993 |
| test | faithful | 0.0095913586 ± 0.00013784493 | 0.0078792879 ± 0.00016405576 | 17.854 ± 0.74764% | 0.00050150575 ± 0.00015807525 |
| test | nll | 0.010610015 ± 0.00050569341 | 0.008774932 ± 0.00022189036 | 17.214 ± 3.0139% | 0.00076399516 ± 0.00022294805 |

With loops on, dense-minus-base test error is **0.00204236 ± 0.000881155** for faithful and **0.00200769 ± 0.000164968** for NLL; dense is worse in all three seeds of each objective. Current-risk minus previous-risk test differences are **0.0000391816 ± 0.0000567699** and **0.00000260643 ± 0.0000192493**, respectively. Fresh scoring does not establish a reliable improvement in this three-seed observed-history diagnostic. The complete validation counterparts and all loop-off contrasts remain in the original report and JSON.

The cap never binds in the measured current base/dense graphs or previous-base scoring graphs; the new and old radius classifiers select the same measured pair universes. Cap inactivity on the 297 test states was already inspected before this protocol (maximum nonself base/dense degrees 22/31), so it is not an independent discovery. Validation confirms the measured cap behavior. These observations do not establish cap inactivity during autonomous rollouts.

Base/dense/random/speed keep their selected nonself pair sets fixed between loop arms. Risk scores and selections change with the loop convention, so risk loop contrasts include both changes. The bridge's random seed material is `[20261005,771,training_seed,split_code,source_index,target_frame]`, shared between loop arms and objectives; this is **a different draw from the original locked random policy**. Cross-study differences cannot be attributed solely to loops. Previous risk uses the preceding observed history; current risk uses the current observed base graph. Neither is the autonomous policy's cached risk from its own preceding selected graph. The bridge globally orders directed edges; the separate native autonomous study preserves native base edges as a prefix before appending optional edges.

All results are exploratory. The exact same already inspected test histories and previously used validation schedule are reused; the first three official test trajectories and historical aggregates were inspected earlier. This is not pristine independent confirmation, a new test population, a significance claim, an autonomous-stability result, or evidence that expanded-edge training support is adequate. Single-call times share work and use fixed case order; they do not establish policy latency or speedup.

## Included and omitted evidence

- `report.md`, `analysis_source.json`, the six `runs/*/{protocol,result,status}.json` triplets and queue records preserve original bytes.
- `graph_convention_bridge.json.gz` decompresses to the exact 53,075,022-byte strict-summary JSON (SHA256 `1623cab0517dd4a20ceb668e322aa98ff064a6dc4e54b62b63d46d7c05ff27d5`). It retains every per-frame derived scalar and all original input-file identities. `results.json` is a lossless projection of its aggregate/seed/trajectory sections; it removes only duplicated per-frame records, input hash dictionaries and full protocol copies, which remain available in the compressed original and run protocols.
- `independent_audit.py` and `graph_bridge_aggregate_audit.json` preserve the separate 5777-check aggregate audit; the strict summary records 851400 geometry, arithmetic, coverage and integrity checks. These are verification counts, not scientific samples. `protocol.md`, `source_freeze.json` and `provenance.json` pin source/configuration/checkpoint/data identities. This curation rechecked current pinned source files and converted manifest bytes; it did not rerun inference or rehash the original dataset/checkpoint payloads.
- `raw_inventory.json` records relative paths, sizes and SHA256 for **all 5129 raw files**, including every omitted output. The omitted 2550 particle NPZ archives, 2550 per-frame raw JSON records and six process logs remain unmodified locally. NPZ holds particle predictions, risks, histories, features and graph arrays. The per-frame JSON contains original detailed case timing and graph records beyond the derived scalar summary. Logs contain process output. No training/validation/test datasets or checkpoints are embedded. Their hashes identify assets; they are not download links or substitutes for missing files. Complete source/geometry reauditing requires the inventoried raw arrays and external data.

`PUBLICATION_MANIFEST.json` and `PUBLICATION_AUDIT.json` check package identity and document exact copied-source relationships. Run `python3 verify_publication.py` from this directory to verify packaged hashes, compressed-source identity, exact aggregate projection and all-case coverage without original data or model inference. Absolute paths in unchanged originals are provenance strings, not portable paths. This is nonanonymous research-fork evidence; anonymity, asset rights and author verification remain separate submission checks.

## Reproduction

The frozen evaluator is `research/graph_convention_bridge.py`; `protocol.md` gives its checkpoint/data/runtime command contract and native identity gate. With all six original raw output directories available, run from the repository root:

```sh
python -m research.summarize_graph_convention_bridge --runs \
  /path/to/bridge/faithful_seed0 /path/to/bridge/faithful_seed1 /path/to/bridge/faithful_seed2 \
  /path/to/bridge/nll_seed0 /path/to/bridge/nll_seed1 /path/to/bridge/nll_seed2 \
  --output-dir /path/to/new-report-directory
```

The strict summary requires a new output directory and all original referenced files; it performs no inference and verifies hashes/coverage. Preserved originals use the recorded local absolute paths, so reproducing these exact identity checks requires that layout or a fresh protocol-governed rerun from available inputs. The separate aggregate-audit script retains its original workspace paths and can be run in the recorded layout. This publication is not a standalone rerun bundle.
