# Native-convention WaterDrop autonomous follow-up

All six fixed 100k models completed the planned 27 trajectories × five policies × 995 forecasts: 810 outcomes. All 405 faithful outcomes completed numerically. Seven NLL seed-2 outcomes hit coordinate guards (base1, dense1, random2, speed2, cachedrisk1); all five complete-cohort NLL full-horizon means remain undefined. No retry or survivor-only headline average was used.

Faithful cached risk minus base has paired full-rollout MSE difference +0.00428683 ±0.00335776; risk minus random is +0.00044705 ±0.00072071, with a sign change across seeds. These three-seed means/sample SDs do not establish statistical significance. Every seed, policy, boundary result, failed prefix and original-convention comparison remains in the exact compressed summary. This exploratory follow-up reuses inspected test sources3–29 and changes native graph conventions jointly; it is not pristine confirmation or an isolated causal self-loop estimate.

The strict source/trace/arithmetic audit passed19,521,125 checks. The complementary independently implemented raw-scalar/paired audit passed1,724,434 checks. Check counts certify saved-artifact consistency, not scientific validity. Both exact result files are gzip-compressed without changing their uncompressed bytes; provenance binds their original hashes. Raw per-step/trace files are omitted with identities in omitted_raw_inventory.json; datasets/checkpoints are not republished here.

The NLL seed-2 process was suspended and resumed at the author's request. Its raw elapsed counters are preserved, including the interruption. They must not be treated as uninterrupted timing or repaired by subtracting a guessed duration. All timing in the raw summary remains fixed-order, policy-geometry-dependent reference-pipeline time including diagnostics, not a fair optimized speedup claim. The pause/resume receipts are retained.

The existing open manuscript was edited in place and compiled successfully with the native editor. All23 preceding tables/title/abstract remain; three appendix tables and one motivated full-rollout paragraph were added. revised_manuscript.tex is the publication snapshot of that same source, not a replacement editor document. Author verification and registration/submission checks remain outstanding.

Regenerate both inserts using only the compressed audited scalars:

```sh
python research/results/native_rollout_completion_20261006/render_tables.py --output-dir /tmp/native-rollout-tables
```

The numerical evaluator/summarizer and protocol remain frozen in research/native_graph_rollout.py, research/summarize_native_graph_rollouts.py and research/protocols/native_graph_rollout_20261005.md. The independent auditor is preserved in the deadline_operations_20261006 family. No new model inference occurs during table regeneration.
