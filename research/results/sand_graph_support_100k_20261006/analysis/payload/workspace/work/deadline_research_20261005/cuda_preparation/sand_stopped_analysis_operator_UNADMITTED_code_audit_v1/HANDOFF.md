# Sand stopped-analysis operator

This is local operational preparation, not admission or execution. Only root may
issue actual closures, the analysis phase, or an operation. `describe` is inert.
The manifest binds the complete local package; the separate independent review
must match the controller and helper bytes before use.

The scientific contract remains six fixed 100k models, 24 original evaluations,
Sand type 6, T320 and H314. No evaluator, cohort, split, checkpoint, policy,
schedule, saved-array worker, summary worker, quota, denominator or random seed
is changed by this package. There is no retraining or numerical retry action.

## Fixed originals and bounds

- A original evaluation tool session: 51595; owner PID 24151, start ticks
  120240947; GPUs 0–3 on yellow-worm-77. Initial children 24164–24167.
- B original evaluation tool session: 12559; owner PID 74093, start ticks
  536613652; GPUs 2–3 on aquamarine-toad-75. Initial children 74097–74098.
  GPUs 0–1 are observed without control.
- A boot: cbe68c69-6a30-47f9-a9f8-b42a80785a1f.
- B boot: 3d254b4e-ede4-492f-839a-5f16b6f3d1b3.
- There is exactly one full 3600-second Sand analysis phase after both original
  evaluation closures, ending no later than 2026-10-07 04:00:00 UTC. Setup,
  inventories, staging, operations, transfers, closure checks and reviews all
  consume that same original hour. No action grants a replacement clock.

The original native qualification is reused only with the exact same hosts,
boots, runtime and frozen native lifecycle definitions. `stage-runtime` checks
them freshly on both hosts. The accepted prior review is
`sand_native_qualification_completed_independent_review_code_audit_v1.json`
(SHA256 74037b1018b31a6f8ca52048d27a1d1098f1ee435f28c6ed7cd0fb100d471076).
No fresh qualification or science is implied by local tests.

## Root invocation pattern

Use the workspace `work/venv/bin/python -B` with this package's
`root_operator.py`. Every action except `describe` requires:

```text
--root-action --state /absolute/canonical/fresh/state/directory
--package-sha256 <the independently reviewed manifest SHA256>
```

Use the same state and manifest for the entire attempt. A suitable fresh state
name is `cuda_preparation/sand_stopped_analysis_released_root_v1` under the local
preparation directory. `prepare` creates only the local binding; it starts no
clock and reads no scientific output. Existing state paths are refused.

1. Run `prepare` once. Wait on each original evaluation tool session to actual
   exit. Preserve its unmodified completed tool response in a local evidence
   file. Then run `record-evaluation-exit --role A` or `B`, supplying the fixed
   `--session-id`, actual `--exit-code`, root's `--observed-utc`, and
   `--evidence-file`. Nonzero evaluation exits are retained and may proceed to
   stopped-evidence collection if native closure can be established.
2. Run `close-evaluation --role A` and then B, each with an explicit fresh
   `--observation-deadline-utc` 20–60 seconds ahead. These short bounded metadata
   observations do not read scalar experiment results or issue analysis time.
   The helper checks the raw original release, union of every registered child
   and outcome, all 16/8 stage ledger entries, original startup identities,
   exact boot, all historical/registered native PID absence, and assigned GPU
   emptiness. A missing native identity is a refusal, not inferred closure.
3. After both closure files exist and are reviewed, run `begin` exactly once.
   It refuses unless the entire 3600-second phase fits before the global stop.
4. Run `stage-runtime --role A` and B in that phase. This stages only exact
   frozen sources and controls, creates/verifies an empty canonical `owners`
   parent, and requires all owner/output leaves to remain fresh.
5. For each operation below, run `run --operation <name>` exactly once. Wait
   on that original root tool to actual completion, preserve its completed
   tool evidence, run `record-original-exit --operation <name> --session-id
   <that original tool session> --exit-code <actual code> --observed-utc <root
   observation> --evidence-file <saved raw evidence>`, then run
   `close-product --operation <name>`. A product cannot be accepted until its
   original tool exits 0, its bounded local transport is reaped without timeout,
   its native timeout/owner/child are absent, and all raw inputs/evidence/output
   bytes and semantics agree.

```text
sand_collect_A
sand_collect_B
sand_saved_A
sand_saved_B
transfer-B
sand_summarize
sand_paired
complete
```

`transfer-B` is an action, not an operation. It transfers only the exact accepted
B collection and saved-array audit JSON to A. Original queue arrays stay on
their original hosts. Either host's collection and saved-array audit may be
completed before moving to the other host; each operation's predecessors are
enforced. Summary and paired audit require the accepted B transfer. `complete`
rechecks all six operation receipts within the same original hour.

## What is checked

Collection is preceded by a whole stopped-tree inventory, an exclusive
collection release, and a second identical whole-tree inventory. Each inventory
binds membership, SHA256 and byte size, including every file rather than only
the numeric results. Original evaluation closure metadata must match. The saved
array audit uses the identical stopped queue. The frozen shared owner rehashes
all source/control/tree bytes, protects tree membership and admits only the exact
six worker argument lists from the unchanged ten-operation map.

Post-exit capture returns the original owner start, child registration, stdout,
stderr, terminal and declared scalar product. The local controller verifies the
full terminal input map and evidence hash set, native closure, absolute guard,
original clock and publication order. It rehashes each decoded collection
metadata/protocol file against its transported snapshot and the CPU release
before invoking the independent product validator.

Publication must remain within the strict same-host invocation/deadline interval.
Only the comparison between host publication time and root's exit observation
uses the already accepted five-second host/root clock error; it grants no extra
runtime or publication time.

The validator preserves all 2432 A and 1216 B cells (3648 total): 1080 rollout,
900 valid same-state, 900 test same-state and 768 clean-validation cells. Failed,
timed out, unavailable and unexecuted outcomes remain in their denominators.
Sparse diagnostic evidence and fixed three-seed aggregation are checked. An
accepted collection with unresolved original input integrity is preserved with
`eligible_for_aggregation: false`; its downstream operations are refused.

## Failure evidence and interpretation

All controls and invocation tags are exclusive. Original `.command.json`,
`.stdout`, `.stderr`, `.external.json`, launch records, exit attestations,
snapshots and captures remain in the local state. Remote owner and worker
outputs are never deleted or replaced. A failure may stop before a scalar product
or native terminal exists; the original tool and transport evidence then remain
authoritative. A tool timeout alone never proves remote native closure.

There is no automatic retry, recovery, resume, overwrite, shortened hour or
deadline-extension action. Any separately justified operational repair must
retain this attempt's artifacts and original clock and be reviewed before use.
Do not repeat a failed invocation under a new state or tag to obtain a preferred
result. If the remaining original hour expires, retain available evidence and
report incomplete analysis.

`accepted_stopped_product` and `analysis_completion.json` certify bounded
operational closure and preserved product contracts. They do not assert all
experiments succeeded, numerical completeness, scientific admission or speedup.
Publication still requires root's completed evidence review and interpretation.

## Local verification

`test_operator.py` uses synthetic metadata and mocked clocks/transports only.
It covers both full host ledgers, quota/error/unexecuted preservation, changed
identities, missing closure, changed byte pins, one fixed analysis hour, all six
frozen command resolutions and fresh owner-parent behavior. `test_review_products.py`
contains the product validator's separate synthetic suite. These tests do not
access models, arrays, live queues, native GPU processes or remote hosts.
