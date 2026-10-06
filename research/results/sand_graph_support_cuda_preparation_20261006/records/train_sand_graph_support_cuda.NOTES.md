# Fresh Sand CUDA graph-support study: prepared entry

Prepared for root review and prospective study selection. **No real model, data or CUDA training was executed here.** This entry does not inherit the old Sand timing release or admit scientific training. The original deterministic Sand trainer, WaterDrop continuation and numerical core remain unchanged.

Files and evidence:

- `train_sand_graph_support_cuda.py`: SHA256 `fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124`.
- `test_train_sand_graph_support_cuda.py`: SHA256 `03c9228031257ac001bd2ed035f0e231248b1fd06358c766d9a3702b8b5b1658`.
- Author run: **16 tests passed in 2.21 seconds**. Independent methods run: **16 passed in 2.14 seconds**. Tests instantiate width-8, one-block CPU networks with synthetic positions, plus mocked CUDA RNG/control calls. They never instantiate the full study model or access real trajectory arrays. Existing pinned GNS docstring escape warnings and a Torch JIT deprecation warning remain visible.
- Independent review: `train_sand_graph_support_cuda_independent_review_final.json`, SHA256 `db3c5bf093ac1738a3110b9759441c3008b0a5bdde913d31452fe05b61f1d133`.
- `train_sand_graph_support_cuda_declared_delta.json` records source-only AST equality for nine inherited data, optimizer, gradient, RNG, pointer and CUDA configuration functions.

## Fixed candidate design

The candidate is **six models: faithful objective × base/mix arms × seeds 0,1,2**, each trained from scratch for exactly **100000 updates**. There is no automatically added six-model NLL cohort. The CLI requires an arm and one of these seeds and rejects a different requested endpoint or objective. A planned `--stop-after` remains an explicitly incomplete infrastructure stop; it does not shorten the declared scientific endpoint.

Architecture, seed initialization, empty Adam, source/data admission, normalization, six-frame histories, batch size 2, noise 6.7e-4 and the 100k learning-rate schedule match the existing deterministic Sand trainer. Width is 128, message-passing depth 10, MLP depth 2, with the original type embedding and scalar faithful variance head. Adam remains unfused, foreach disabled, without clipping or weight decay. Strict CUDA determinism, environment requirement, Torch 2.13.0+cu129, GB200 restriction and TF32/AMP/compile/DDP settings are unchanged.

Both arms call the byte-pinned `research.faithful_graph_support.forward_batch` and `append_optional_edges` functions without editing them. They preserve the exact native strict-radius .015, capped-128, self-candidate directed base prefix. Optional pairs come from the full uncapped strict geometric annulus at radius 1.267r, excluding the full strict-r graph. Short-range edges omitted by the native cap never become optional candidates.

For each noisy training example independently, mix uses a Bernoulli .5 coin. On success, it appends exactly `floor(.25 * annulus_count)` uniformly selected nonself unordered pairs in both directions, ordered as a separate suffix. It never reapplies the cap, changes the mandatory prefix or connects separate examples. Base appends nothing. Zero-budget cases return the original native edge tensor and native features. Both arms pay the same candidate/audit helper cost.

The existing graph RNG rule is reused at fresh training steps **0..99999**: coin material `[20261005, seed, completed_step, example_slot, 4409]` and pair material with final value `5501`. Arm identity is absent from graph, frame and noise RNG material. The original host frame/noise helpers are unchanged. Selection sees only the noisy observed input; targets enter only the unchanged inverse-decoder target computation.

This intervention tests whether expanded-graph training support changes allocation behavior. It does not make the residual variance head an action-benefit estimator and does not guarantee risk ranking will improve. All policy outcomes and paired arm differences remain necessary.

## Provenance and outputs

The distinct schema is `adaptgns_sand_graph_support_cuda_training_v1`; checkpoint key is `cuda_sand_graph_support_schema`, and training metadata uses `cuda_sand_graph_support_run`. Configuration records the arm, fresh initialization, full graph-exposure recipe and source hashes. Old Sand, WaterDrop full-training and WaterDrop continuation schema fields are rejected. Resume requires this exact configuration and source/arm identity; no parent checkpoint or cross-lineage reuse is supported.

Every completed update retains frame IDs, noise hash and the existing per-example graph ledger: exposure coin/material, native/cap/self counts, annulus budget, selected count and graph/noisy-state hashes. Checkpoints contain this ledger, optimizer, model, CPU/selected-CUDA RNG, schema/configuration and update count. Ledger ordering, exact budgets, arm flags, RNG material and hashes are checked on save/restore. Root should compare initial state tensors and all host schedule records across paired arms; checkpoint files themselves differ because their arm metadata differs.

Default checkpoint interval is 10000 and logging interval 100. Checkpoint overwrite is refused. Existing partial files block a resume write; JSON publication uses exclusive temporary files. Each attempt has its own status; unsuccessful context/history/raw model and optimizer are preserved without promoting the rejected state. Final source/protocol/manifest/admission/structural hashes and numeric data bytes are rechecked.

The per-update ledger and full-Adam guards, repeated ledger validation and checkpoint serialization are deliberate measured overhead. The earlier base-only faithful/NLL capacity probe cannot establish this trainer's throughput. Benchmark this new entry on its own fixed paired base/mix schedule, with no scientific checkpoint promotion, and include full model/evaluation/diagnostic costs before launch.

## Required root follow-through

1. Choose and freeze this candidate's scientific protocol, complete evaluation/summary entries and exact workload before outcomes. Existing final Sand helpers assume faithful/NLL cohort identity and the old training schema; they must not accept these checkpoints unchanged.
2. Validate the unchanged native-base extraction and faithful gradient semantics **within CUDA** on admitted batches; verify mixed-graph prefix/budget/partition behavior and same-device checkpoint/optimizer/RNG replay. The tiny CPU tests are preparation evidence, not an actual-CUDA pass. Earlier failed CPU/CUDA equivalence reports remain failed.
3. Measure sustained paired base/mix capacity and checkpoint/ledger I/O, then forecast all six 100k models, all declared policies, diagnostics, verification and writing reserve. Additional GPU count alone does not establish the forecast.
4. Issue a separate launch release only after those checks. Final-only validation and reserved-test access remain separately gated; retain unfavorable and incomplete outcomes without model or seed replacement.

Example argument shape, **not a launch authorization**:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_sand_graph_support_cuda.py --execute
  --repo <frozen-repository> --train-manifest <admitted-train.json>
  --admission <train-admission.json> --structural-report <structural-report.json>
  --protocol <new-frozen-scientific-protocol.md> --output-dir <fresh-study-directory>
  --objective faithful --arm mix --seed 0 --updates 100000
  --cuda-index 0 --threads 2 --checkpoint-every 10000 --log-every 100
```

The description-only default imports standard-library modules only. No source changes in this candidate alter a frozen experiment or relabel its checkpoints.
