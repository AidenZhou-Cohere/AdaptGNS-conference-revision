# Exploratory graph-convention bridge: manuscript artifacts

These figures and LaTeX inserts render the separate [complete graph bridge](../graph_convention_bridge_20261005/README.md), with three paired seeds, both objectives and both observed-history splits. They do not replace the original no-loop results or report the still-running native autonomous study. All reported contrasts are exploratory, not independent confirmation or significance tests.

Reproduce from the repository root, choosing a new output directory:

```sh
python -m research.render_graph_bridge_findings \
  --summary research/results/graph_convention_bridge_20261005/graph_convention_bridge.json.gz \
  --audit research/results/graph_convention_bridge_20261005/graph_bridge_aggregate_audit.json \
  --output-dir /path/to/new-bridge-render
```

The renderer rejects a changed uncompressed summary hash and checks each headline against all seed values. `render_manifest.json` pins the producer and original rendered outputs. PNG/PDF metadata may differ across render environments; the LaTeX inserts reproduce byte-for-byte. Plot values are test seed means, not individual particles; line segments pair loop arms within seed. All MSE table/plot values are normalized acceleration coordinate error, with the displayed scaling explicit. All16 cases and37 contrasts remain in the source publication, including caps, adverse signs and validation counterparts.

The current standalone manuscript compiles natively with the eight-page main-text assertion; the15-page evidence report's changed pages were visually inspected. Native compiler output does not export the submission PDF. Author review/export and submission checks remain outstanding. The local initial renderer missing-key attempt is preserved; no numerical source/output changed.

The six sequential model processes used **693.156 seconds** (11.55 minutes), with a693.177-second first-start/last-completion timestamp envelope. `measured_runtime.json` preserves every model duration and timing scope. This includes loading, inference, diagnostics and writing but excludes development, separate audits and document work; it is not a policy latency or speedup measurement.
