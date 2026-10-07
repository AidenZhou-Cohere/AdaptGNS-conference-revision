# Distinct metadata observation attempt 2

Root operational review is required before execution. This wrapper changes only
three local output path constants while executing the exact original functions
from scientific package `1dee2ec7c67639fab665e9fe88ce2cd6fb5862ac9eb9289fafb29446c9165e62`.
The original package and all first-attempt artifacts remain unchanged.

Attempt 1 observer 76617 and metadata proxy 94805 both actually completed with
exit 0. Its host finished timestamp was 2026-10-06T23:24:52.247114+00:00; the actual
root clock sample was 2026-10-06 23:25:02 UTC. The 9.752886-second separation fails
the unchanged 5-second clock-pair condition. Root did not invoke begin, issue a
phase, stage remotely, or start a scientific worker. This is a distinct metadata
prerequisite attempt before any D3 observed-history phase, not a worker retry.

Only these runtime module constants differ:

- `prephase_sources.PROXY_OUTPUT`: `PREP/goop3d_observed_fresh_route_attempt2_root_v1`
- `prephase_sources.OBSERVER_OUTPUT`: `PREP/goop3d_observed_fresh_metadata_attempt2_root_v1`
- `root_fresh_begin.PREP`: `PREP/goop3d_observed_fresh_entry_attempt2_root_v1`

The last constant is used only for the original entry assembler's output; its
final output is the nested directory
`PREP/goop3d_observed_fresh_entry_attempt2_root_v1/goop3d_observed_fresh_entry_root_v1`.
Original `root_prepare.PREP`, package `HERE`, phase `STATE`, source payload,
scientific sources, source hashes, 13+4 staging, and generated startup commands
are untouched. Imports resolve to the frozen package. The original owner is not
imported by preparation or synthetic checks.

The observer remains the exact original 42-PID metadata observer with local 35,
remote outer 20, and internal 12-second limits. The original metadata proxy is
limited to 40 seconds. Host/root observations must still each be at most 60
seconds old with at most 5 seconds difference. A full 3600-second phase must fit
before 2026-10-07T01:00:00+00:00. T0 is still the actual new host observation's
finished timestamp; it cannot be shifted. All remote staging and pipeline launch
remain after T0 and before T0+100. Every later original deadline is unchanged.

Root sequence (substitute the reviewed wrapper manifest SHA for WRAPPER):

1. Run local-only `work/venv/bin/python -B WRAPPER_DIR/attempt2.py plan --root-attempt2 --wrapper-sha256 WRAPPER`.
   It validates and reuses the existing local source payload and emits exact
   metadata proxy and observation shell commands. It creates no phase and does
   no remote staging. It refuses existing attempt2 paths or an existing phase.
2. Root starts the emitted 40-second metadata proxy. Run its emitted observer
   command and poll its original tool to actual completion, then sample the
   independent root clock immediately in the same `functions.exec` cell. Do not
   wait for proxy completion before sampling the clock. Preserve the actual raw
   observer launch/completion and actual clock response under distinct attempt2
   paths; do not construct a successful exit or timestamp.
3. Wait for the metadata proxy's actual original exit, preserving its raw tool
   envelope plus the exact terminal path and hash. Root then invokes:

   `work/venv/bin/python -B WRAPPER_DIR/attempt2.py begin --root-attempt2 --wrapper-sha256 WRAPPER --root-begin --package-sha256 1dee2ec7c67639fab665e9fe88ce2cd6fb5862ac9eb9289fafb29446c9165e62 --observer-tools OBSERVER_TOOLS --observer-tools-sha256 OBSERVER_TOOLS_SHA --observation-proxy-tools PROXY_TOOLS --observation-proxy-tools-sha256 PROXY_TOOLS_SHA --root-clock-json ROOT_CLOCK --root-clock-json-sha256 ROOT_CLOCK_SHA --source-payload SOURCE_PAYLOAD --source-payload-sha256 SOURCE_PAYLOAD_SHA --observer-directory PREP/goop3d_observed_fresh_metadata_attempt2_root_v1 --phase-proxy-directory FRESH_PHASE_PROXY_DIRECTORY`

   Use absolute paths. The wrapper calls unchanged `root_fresh_begin.main`, which
   checks actual completed tool evidence and local native absence, and calls the
   unchanged phase issuer and all its timing checks. No waiver or clock reset is
   added. A failure remains a failure.
4. Execute only the original phase proxy and combined stage/pipeline commands
   emitted by the original issuer. Follow the frozen scientific package's
   `HANDOFF.md` for collection, all47 native closure and final admission.

Source checks and nine tests are synthetic/local only. They establish no fresh
metadata observation, phase, startup, remote closure, or scientific result.
