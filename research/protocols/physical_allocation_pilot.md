# Cheap physical allocation controls: separate exploratory compact pilot

Specified October 4, 2026, after the original compact-pilot outcomes and the physical-descriptor correlation analysis were inspected, before predictions under these new policies. This is an exploratory follow-up, not independent confirmation. It does not modify the running full-architecture training, the fixed 100,000-update checkpoints, or either locked full-model evaluation. No full-model checkpoint or reserved full-test source is opened.

## Population and fixed predictors

Use every existing compact pilot: faithful, NLL and beta-NLL, seeds 0–4, width 48/depth 3, final 3,000-update checkpoints. Use the identical 36 validation and 36 already inspected test frames/model: three trajectories with twelve frames each. Reproduce load_frames sampling with seeds 811 and 812 and require exact equality with the saved pilot protocol and control frame lists. Do not train, fit a score, select a checkpoint, tune a radius or select an objective based on these results.

Before loading models or numeric data, verify physical_allocation_inputs.json: hashes of the original saved summaries, pilot protocol, every checkpoint and original control record, and metadata/validation-pilot/test-pilot source files. Check the pilot source hash against its original training protocol. Load known checkpoints on CPU using restricted weights-only loading with the known TorchVersion metadata allowlist, and require exact objective, seed, protocol and architecture/state identities. Preserve source, configuration, input and software hashes with all outputs.

## Graph and scores

Use the original pilot candidate graph, with radius 0.015, expanded radius factor 1.267, symmetric undirected pairs, no self loops and the original pair ordering. Preserve its numerical convention: cKDTree candidate search followed by the original float32 squared-distance <= base-radius test. This is intentionally distinct from the strict float64 neighborhood in the earlier physical-correlation diagnostic. Do not silently change the graph builder or conflate those neighborhood counts.

At target t, use only the observed displacement u_i = float64(x_i[t-1]) - float64(x_i[t-2]). The supplied mandatory base pairs define both new node scores:

- inverse_count25: s_i = -d_i, where d_i is the number of mandatory base neighbors. This tests preferential expansion near sparse sampling; it is not calibrated density or free-surface detection.
- velocity_rms25: s_i = sqrt(sum_{j in base_neighbors(i)} ||u_j-u_i||² / d_i). For d_i=0, set this score to zero by a predeclared empty-neighborhood convention and explicitly record the isolated-particle count. Do not impute gradients or drop particles using the earlier strain-fit mask.

For each score, rank optional pairs using max(s_i,s_j), retain exactly K=floor(0.25 × number of optional pairs), and always retain every mandatory pair. Use the original stable lexicographic particle-ID tie rule. Record K, base/optional counts, selected optional pairs, score ties at the selection boundary (all tied pairs and selected slots), and pair overlap with each replayed control. Scores may be permutation equivariant while the ID tie rule is not; record ties rather than asserting equivariant selection. No 'add none', threshold search or variable budget is introduced.

## Paired controls and outcomes

Replay all six original policies on the same model/state: base, dense, random25, speed25, current_risk25 and lagged_base_risk25. Reset the original random selector RNG to 91300+seed independently for each split. Current risk comes from the observed base graph; previous risk comes from the preceding observed base graph. The latter is teacher-forced and is not autonomous cached-own-graph control. Reusing the already computed base mean is allowed.

Before accepting a frame, require identical original-control edge counts and coordinate MSE within the already used replay tolerance rtol=2e-5, atol=1e-7. A mismatch stops this experiment; do not widen tolerances after outcomes. Retain replay differences. Check finite predictions, positive finite risk and nonnegative finite errors; preserve any numerical failure, invalid score, corrupt join or incomplete run.

The primary measurement is one-step normalized-acceleration coordinate MSE over every particle, with the same target/normalization as the compact pilot. Also retain every particle's vector squared error. Average particles within each frame, then all twelve frames within a trajectory, the three trajectories within a seed, and all five seeds. Report each seed, its paired new-policy-minus-control difference and percent difference, and five-seed means/sample SD. Percent differences require a nonzero control mean. A required failed/missing frame or seed makes the corresponding unconditional aggregate null; do not substitute a survivor mean. Report both splits and every comparison, including unfavorable outcomes, with no p-values or best-case selection.

This tests these two selected-edge interventions on observed histories. It does not establish rollout stability, benefits in the full architecture, physical causality, or deployment speedup. Positive whole-dense benefit is neither necessary nor sufficient for a useful selected subset.

## Operational limits and preservation

Use one CPU process with one Torch/BLAS thread and no GPU. Do not start a new frame once 300 seconds of analysis wall time have elapsed. Finish/preserve any frame already in progress and its output flush; total process time may therefore exceed 300 seconds. This is an operational start limit, not a shorter scientific sample. Preserve partial records and raw outputs and label incomplete groups explicitly. Use a fresh output directory; never overwrite or automatically retry an unsuccessful attempt. No training process is launched, stopped or modified.

The saved pilot's forward timings suggest seconds-to-minutes of work, making this feasible during the ongoing local Metal training. This measured feasibility replaces the earlier prepare-only scheduling assumption, not the scientific protocol of the full experiment. Record actual total time and component timing scope. Shared candidate construction, reused base predictions, score computation and concurrent training mean these timings are diagnostic/operational measurements, not a fair optimized policy-speed frontier.

Before real inference, review the implementation and synthetic tests, then commit the new protocol, input manifest and numerical source. Keep raw per-particle arrays and per-frame graph choices local under work/. Curate complete scalar summaries, code, verification and all failure records into the research fork after review.
