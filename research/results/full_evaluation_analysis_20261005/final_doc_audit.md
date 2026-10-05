# Final document correspondence audit

Passed: **True**. 658 focused correspondence checks and 19 final byte-stability checks. No remaining issues.

Current manuscript SHA256: `999837597a585169431565b4a549d5bda7cc87b438d3f89b2f49188cc3c3cab7`.

All ten new tables agree with audited rollout/same-state summaries at their displayed rounding, including failed/undefined entries, seed values, paired differences, boundary references and milliseconds. The five current result summaries reproduce the checked quantitative anchors and explicit limits.

Main title and abstract are unchanged from the pre-integration body. All 4 existing tabular blocks and 12 earlier body labels remain; 36 current labels are unique and references resolve. Main and appendix inserts appear verbatim in the standalone source.

The public renderer checks both pinned summary hashes before writing. Fresh rendering matches current insert and manifest bytes. Main SHA: `29090bdea3f5faed03f4357fe6cd35b2039cebc3a304e9d57285358a866239f8`; appendix SHA: `daba11cfb914a41028f598317042266a8cfe29ab30c843c18525238f226f323b`.

Resolved findings:

- outputs/README.md: Current locked-test evaluation described as running. Now explicitly complete; all results linked.
- outputs/README.md: Refinement paragraph left full evaluation pending without supersession. Now states later full-model evaluation complete; author verification remains pending.
- outputs/submission_checklist.md: Completed comparative experiments described generally as a small convenience sample. Small-sample scope now specifically applies to compact-pilot experiments.
- outputs/AdaptGNS/research/results/full_evaluation_analysis_20261005/paper_results_generation.json: Transient stale appendix output digest during publication integration. Current public/local generated manifests and fresh renderer outputs match.
- work/full_evaluation_main.tex: Final checkpoints fixed before test access was broader than evidence permits because first three test trajectories were previously inspected. Now explicitly before the locked test evaluation, with prior inspection disclosure retained.

Full-model outcomes, compact pilot and historical evidence remain distinct. All eight failures, repeat variation and unsuccessful audit history remain documented. No broad superiority, optimized speedup, physical validity or conference readiness is asserted. Historical progress notes are clearly dated and superseded.

An initial scalar-check process printed its entire check ledger; tool output was truncated and orchestration JSON parsing failed. The process did not edit artifacts; no numerical mismatch was observed in that failed transport attempt. The focused check was repeated with compact output and saved as final_doc_numeric_check.json.

This source-level review does not certify exported PDF layout, the registered abstract, human verification, rights or anonymous packaging. Parent will refresh the derived-publication manifest after adding this audit.
