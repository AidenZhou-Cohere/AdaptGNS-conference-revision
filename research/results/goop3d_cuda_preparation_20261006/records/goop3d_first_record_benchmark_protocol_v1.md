# Goop-3D fixed first-record capacity benchmark

This is a bounded timing/memory check on the already inspected first training record. It does not admit full-dataset training, evaluate efficacy, read test data, select a checkpoint or change any frozen 2D source. Root owns launch, current process/physical-GPU identity checks and the outer timeout.

The reviewed report SHA `c003d147b93eb0eb2cebafe7c9a79ac4179d7224fd41d139e1719f5ceefedd78` fixes source record `train:000000`, N9271/T301/D3. The benchmark verifies its three numeric array hashes, exact Goop-3D metadata SHA `727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55`, and pinned official parser SHA `5868022f76eaf15b2626125bdaa3c973ccaf2dfc0b0ec50d9e05d0c3d973e149`. Context remains preserved and excluded under the released parser rule because metadata lacks `context_mean`; this does not replace full-source auxiliary admission. All source/input hashes are rechecked at completion.

Two copies of target6's six observed frames form batch2. Labels and one CPU keyed noise realization are fixed for both cases and every update. This repeats one state intentionally; it does not sample the complete training distribution. The full original128-wide/10-block/2-MLP simulator has37 node features,4 edge features and3 acceleration outputs, radius0.025 and noise6.7e-4. Correct faithful loss uses summed3D squared error, detached-residual likelihood with coefficient1.5 on log(q), and detached variance features. Coordinate variance isq and vector risk is3q. Adam uses1e-4, betas(.9,.999), eps1e-8, no weight decay, foreach=False/fused=False. There is no AMP, TF32, compile, CPU fallback or parameter tuning.

Run fresh identical seed0 models sequentially for native `base` and benchmark-only `expanded25`. Base preserves the strict-radius directed cap128/self graph. The expanded case preserves that exact native prefix and forces both examples to append symmetric random floor(0.25×annulus pairs) edges from the uncapped strict radius1.267×0.025 annulus. No appended edge is recapped. Its fixed keyed pair selection never touches Torch RNG. This is the cost of full exposure on this state, not the probabilistic mixture distribution or a proven upper bound for unseen states. Pair/edge counts in every row determine whether optional-edge coverage was present; an empty annulus cannot establish expanded-graph throughput.

Each case receives1 warmup and3 measured faithful Adam updates,8 optimizer updates total. Report every attempted and completed update, finite checks, loss and dimensional diagnostics, exact graph ledger, fresh initialization hash, synchronized wall time and peak allocated/reserved CUDA bytes. Timed updates include fixed host tensor transfer, candidate precheck/native/optional graph construction, model forward, loss, backward, Adam, finite checks, dimensional diagnostic scalars and CUDA synchronization. They exclude dataset verification, creation of the fixed batch/noise, model initialization, report writes and final hashes. Loss/diagnostics are operational observations, not evidence of prediction quality. No checkpoint is published or eligible for promotion.

This separate 3D capacity regime uses a prospective2,000,000 candidate-pair limit per example and5,000,000 directed-edge limit per batch. The candidate bound is checked before native neighbor lists are materialized. These differ from the frozen2D100,000-pair guard and must never be silently compared as identical resource conditions. Any OOM, nonfinite output/gradient/parameter, graph guard or timeout remains a failed attempt with its existing report/logs. Do not retry or change guards merely to obtain a passing benchmark.

`benchmark_goop3d_first_record.py` SHA `ece79713bb3f029daad415039f633d412ca3093122e343fd9b31629f81917c55` pins `goop3d_graph_support.py` SHA `7d43fe7d06ac450b6b9031181902cfc6d3d9754af3a26e17e96fd46c4a1bf64f` and the five unchanged simulator/loss sources. Fourteen tiny CPU/synthetic tests passed, including exact base-forward/native graph agreement for the full architecture in3D, strict-radius/cap/tie ordering, exact optional budgets, candidate guards before neighbor allocation, the1.5 likelihood coefficient,3q vector risk and finite expanded Adam. Warnings are retained in `goop3d_first_record_benchmark_synthetic_tests_attempt1.log`. No official arrays or CUDA were used by the preparing/reviewing agents.

## Root launch

Copy the two scripts byte-for-byte into the existing second-VM repository's `cuda_preparation/`. Verify current process state, free memory, the intended physical GPU UUID, source hashes and absence of the fresh output below. The first-record directory has already been copied by root. Metadata is the checked first object from the ongoing train/valid acquisition; the official parser source is already used by Goop2D preparation. Use the actual selected idle GPU UUID in the one placeholder below, with matching CUDA index. Save stdout/stderr and process identity outside the output directory.

```sh
timeout --signal=TERM --kill-after=30s 600s \
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  /root/repos/AdaptGNS-cuda-20261006/.venv/bin/python -B -u \
  /root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_goop3d_first_record.py \
  --execute \
  --repo /root/repos/AdaptGNS-cuda-20261006 \
  --inspection-dir /root/repos/AdaptGNS-cuda-20261006/goop3d_first_record_20261006_v1 \
  --metadata /root/repos/AdaptGNS-cuda-20261006/goop3d_source_train_valid_20261006_v1/metadata.json \
  --official-reading-utils /root/repos/AdaptGNS-cuda-20261006/cuda_preparation/goop_context_semantics_sources/reading_utils.py \
  --output-dir /root/repos/AdaptGNS-cuda-20261006/goop3d_first_record_capacity_20261006_v1 \
  --cuda-index 0 --gpu-uuid ROOT_SELECTED_PHYSICAL_GPU_UUID --threads 2
```

Leave CUDA_VISIBLE_DEVICES unset. The worker's bound is8 updates; the GNU timeout covers the whole invocation, sends TERM at600 seconds and KILL30 seconds later. The worker catches TERM for a failure report when control returns to Python. A hard kill or host loss may leave the last-progress/temporary report; retain and inspect all output. Root must verify worker exit and device release before launching the exclusive Goop2D scientific supervisor. Completed status is `complete_fixed_state_capacity_only`, never full3D training admission. Actual full-cohort counts, particle extremes and structural integrity still determine whether later3D training is feasible.
