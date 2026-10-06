# Sand evaluation quota review, prospective version 2

Independent statistical/planning review, 2026-10-06. This is not a source freeze,
scientific release, or execution authorization. Preserve v1, its failures and
its undefined complete-horizon forecast. A separate v2 cost basis may use an
enforced allocation; it must never claim that the v1 timing requirement passed.

The proposed six A4/B2 streams each have four ordered invocation allowances:
full-rollout test 7200 s; same-state validation 1800 s; same-state test 1800 s;
clean validation 900 s. This is 11700 s per concurrently executing model stream,
not 70200 s elapsed and not an eight-GPU speedup. Reserve 15 s owned-process
cleanup per stage (60 s per stream), plus explicitly bounded outer admission,
cohort/load validation, result auditing and analysis. Whole-process clocks must
cover imports, checkpoint/source checks, inference, transfers, compression,
fsync and publication. A Python alarm only around inference is insufficient.
The parent needs an independent monotonic deadline, exact PID/start/argv/parent
ownership, bounded termination/reap and no retry. Do not start a next stage if
owned cleanup is unresolved or remaining deadline fit is lost. Host exclusivity
and all other frozen process protections remain in force.

The quotas are prospective scheduling allocations, not estimates of completing
all forecasts and not empirical worst-case guarantees. 7200/180=40 seconds per
planned full rollout on average, before invocation costs; 1800/150=12 seconds
per common-state frame; 900/128=7.03125 seconds per clean frame. These are useful
budget descriptions only. Do not derive an observed full-horizon runtime from
failed 512-update prefixes, repeat probes until stable, choose the lone passing
model, or increase quotas in response to favorable scientific outcomes.

Retain the exact scientific workload: six final 100k models, six policies,
30 test sources and H314 give 1080 required full outcomes. Each model has
150 validation and 150 test common-state frames, plus128 clean-validation
frames:124704 required diagnostic forwards across the cohort. Quota expiration
can make the study incomplete; it does not reduce its denominator or redefine
the prescribed horizon. No timing checkpoint is promoted to a scientific model.

Every planned item needs one accounting status: committed complete-H result;
committed computational/numerical guard failure (with category, phase and
accepted prefix); interrupted current item with no returned committed result;
or never started. A timeout is an execution-budget event, not a coordinate,
nonfinite, pair-cap or parity failure. If exact current work cannot be recovered,
report it as unknown/uncommitted, never invent a prefix. A completed item left
only in a partial file is not silently promoted. Preserve temporary outputs,
process return/signal evidence, JSON/array hashes and all unsuccessful attempts.
Implementation/parity failures still require review before more inference.

Because the unchanged evaluator uses fixed source/policy order, quotas may
select which items finish. All-required full-horizon MSE and paired contrasts
remain null when any required observation is absent/failed. Do not compare
survivors as though they were the declared population, count never-started
items as model failures, or use the incomplete512 outcomes as scientific
performance evidence. Report actual guard rates among all attempted declared
items with missing executions separately. Per-seed risk-minus-random and the
mix-minus-base interaction remain the estimands; n=3 training-seed pairs,
not particles/frames/trajectories as independent training replications.

Saved Sand training arithmetic is 26412.735620 s core forecast plus991.935972 s
terminal-ledger reserve. Adding11700 s evaluation,60 s cleanup,2700 s separate
cohort/load reserve and3600 s aggregation gives45464.671592 s (12 h37 m45 s),
before new acquisition/launch delay. At04:29UTC this nominal plan finishes about
17:06:45UTC, leaving7 h53 m15 s until the October7 01:00UTC compute/analysis
end. Training and reserve factors remain engineering allowances; only reviewed
external enforcement gives an operational time limit, not scientific completion.

Consequently, two similarly expensive full studies do not automatically fit
sequentially under current whole-host exclusivity. Honor the current Goop2D
priority, measure its own six-model capacity without test access, and allocate
one complete study first. Admit Goop3D or Sand additionally only if its entire
remaining source/numerical/training/evaluation/analysis path fits. Do not count
hostB GPUs2/3 as usable concurrently while an unchanged Sand scientific
supervisor rejects unrelated whole-host GPU processes. The October7 08:00UTC
cutoff remains writing reserve, not extra training time.
