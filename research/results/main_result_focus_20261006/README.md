# Main result presentation

This source revision promotes the existing paired Goop exposure/placement figure to the main paper and shows every policy under both training arms at the full H395 horizon, including all three failures. Secondary @200 values remain in the appendix. The complete conditional allocation derivation is moved intact to the appendix, with a qualified main-text summary.

No experiment, result or hypothesis selection was rerun. The title and abstract remain exactly unchanged pending the author's registration comparison. The source passed independent scientific/preservation review and native compilation with the existing eight-page main-text assertion. Final exported-manuscript PDF inspection is still pending.

`revised_manuscript.tex` is the complete standalone source for the native editor. `canonical/` contains all inputs needed for the established builder. To reproduce the standalone source from this package, run `mkdir -p outputs` followed by `python3 -B work/build_manuscript.py` from the `canonical` directory; the result in `canonical/outputs/revised_manuscript.tex` must match the top-level source byte-for-byte. Compilation uses the native editor and needs no terminal TeX installation.

The independent review binds the already admitted Goop summary in `research/results/goop2d_graph_exposure_100k_20261006/paired_scalar_summary.json.gz` (SHA-256 bb3c3f54cdeaf2b1198f49b8437dc711d80ce5e8347ab70363558ab515852978). Earlier evidence families remain unchanged. This package is research source and review material, not a submitted PDF.
