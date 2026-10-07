# Original final-admission critical window

This is a separate inert local helper for root review. It does not modify the
scientific package, original phase, deadlines, authoritative command plan, or
validator. Root alone may execute it after independent review.

Authoritative plan: `../goop3d_original_postpipeline_command_plan_transition_v1.json`,
SHA `564b898c97a0e3efb00e49ca112530ff37548ad4bd61c6c66e15cf43002720e6`.
Scientific package SHA:
`1dee2ec7c67639fab665e9fe88ce2cd6fb5862ac9eb9289fafb29446c9165e62`.
Original pipeline session: 33822. Original phase-route session: 78014.

The route's scheduled end is **2026-10-07T00:30:18.235220+00:00**. Final admission
must finish strictly before **2026-10-07T00:30:58.235220+00:00**. The original
phase ends at 00:31:18.235220. These are fixed existing bounds, not new grants.

## Source-only feasibility review

No mandatory wait or new remote operation in the final validator inherently
consumes the 40-second interval. It constructs `ObservedPhase` without calling
the expired route's `remaining()` method, so it can validate after route end.
It hashes the 67 frozen payloads; reads all ten collected members; parses eight
JSON members; reconstructs arithmetic from 2,568 observed rows and accounting
for all 4,728 cells; verifies owner/tool/all47-process evidence; performs one
local `ps` with a five-second timeout; hashes the ten files again; and writes
and fsyncs the admission. It does not read model arrays or use the network.

Source review does not prove a 40-second runtime. Each member's 256-MiB cap
allows a theoretical 2.5-GiB raw envelope before JSON expansion. Actual collected
sizes will qualify feasibility. JSON parsing and the pure arithmetic checker
have no internal deadline callbacks; subsequent checks, including publication,
still reject completion after +3580. The route may take time to tear down after
its scheduled end: a 0.2-second listener timeout and up to eight 0.1-second
thread joins are visible in source, plus scheduling and file-I/O latency.
No exact exit time or successful final result is established by this review.

The continuation that obtains successful collection and closure must stay active
through the original route completion. Do not defer this critical window to
another heartbeat. Keep actual completion retrieval, final evidence saving,
and helper execution in one `functions.exec` cell without a model turn between
them. Do not pre-run the validator using an invented proxy exit, skip its final
reread, retry a failed stage, reset a clock, or extend a deadline.

## Preassembly after genuine collection and closure

Use absolute paths and the reviewed helper manifest SHA, denoted HELPER below.
Preserve raw original pipeline, collection, and postcollection tool envelopes.
The helper's prepare mode requires actual successful collection and closure
responses, reads their bound local receipts and transport metadata, and checks
the preserved original pipeline33822 launch and actual completion. It reads no
scientific product contents. No actual preassembly has been fabricated here.

`work/venv/bin/python -B HELPER_DIR/final_entry.py prepare --root-final-entry --helper-sha256 HELPER --collection-tools ACTUAL_COLLECTION_TOOLS --collection-tools-sha256 ACTUAL_COLLECTION_TOOLS_SHA --closure-tools ACTUAL_CLOSURE_TOOLS --closure-tools-sha256 ACTUAL_CLOSURE_TOOLS_SHA --proxy-tools FRESH_FUTURE_PROXY_TOOLS_PATH --output FRESH_PREASSEMBLY_PATH`

Prepare rejects an existing future proxy-tools path or preassembly output. It
fills the authoritative final argv with genuine collection/closure paths and
hashes and a chosen fresh proxy-tools path. Exactly two placeholders remain:
`phase_proxy_tools_sha256` and `phase_proxy_terminal_sha256`. Preserve its actual
returned preassembly SHA for finalize.

## One root critical cell after actual original78014 exit

1. Be actively awaiting original session78014 near the scheduled route end.
   Obtain its genuine tool completion; require exit0 and no running session_id.
   Do not signal or relaunch the original proxy or any scientific operation.
2. In that same `functions.exec` cell, save the preselected proxy-tools file as
   exactly `{ "launch": ORIGINAL_PINNED_LAUNCH_OBJECT, "completion": ACTUAL_RAW_COMPLETION_OBJECT }`.
   The original launch file/path/SHA are already bound by authoritative plan564b898c.
   Preserve all raw completion fields. The terminal remains the original
   route-directory `terminal.json`; do not construct or modify it.
3. Immediately invoke:

   `work/venv/bin/python -B HELPER_DIR/final_entry.py finalize --root-final-entry --helper-sha256 HELPER --preassembly ACTUAL_PREASSEMBLY_PATH --preassembly-sha256 ACTUAL_PREASSEMBLY_SHA --proxy-tools ACTUAL_PRESELECTED_PROXY_TOOLS_PATH`

   The helper reconstructs and compares every preassembled metadata binding,
   verifies the exact original78014 launch and successful genuine completion,
   hashes the actual tool bundle and terminal, checks original route metadata,
   and fills only the last two hashes. It calls `os.execv` with the original
   validator and authoritative argv. The original validator performs every
   semantic, native, hash, and final-deadline check and remains the only author
   of `product_admission.json`.
4. Preserve the original final tool's actual exit and returned admission SHA.
   A receipt file without successful tool completion is not a successful final
   admission, including if fsync/write occurred before a later deadline error.

The helper replaces itself instead of leaving a parent wrapper. Its filename
and argv do not contain the exact route program basename used by the original
local native-closure filter. Do not surround it with a waiting shell/Python
wrapper whose full argv contains both that basename and the exact route output
directory; such a process would correctly fail the original local absence check.

The helper is inert without `--root-final-entry`. Neither import nor tests call
clocks, processes, network, the original validator, or scientific products.
