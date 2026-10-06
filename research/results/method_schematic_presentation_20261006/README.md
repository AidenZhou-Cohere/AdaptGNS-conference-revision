# Method illustration and manuscript build synchronization

The paper now illustrates the cached graph controller with a fixed particle state: four mandatory pairs, six candidate pairs and one selected addition. This is an explanatory drawing, not an experiment. The elementary residual-second-moment derivation is preserved verbatim in the appendix so the main text can move directly from the residual signal to the graph action. The title, abstract and all 27 existing tables remain unchanged. No training source, model, result or protocol changed.

The [standalone manuscript](workspace/outputs/revised_manuscript.tex) compiled successfully with the native editor, including its eight-page main-text assertion. The geometry and exact budget were verified from all 28 pair distances. The final exported PDF still requires visual author inspection; compilation alone does not certify layout.

A reconstruction check also found that the older manuscript builder omitted already-published native-rollout and self-loop correlation additions. The current main/appendix components now preserve those additions, and `work/build_manuscript.py` reproduces the open source byte-for-byte. The initial mismatch is retained in `canonical_build_difference_v1.diff`; the final diff is empty. This synchronization adds no new numerical evidence.

To reconstruct, run `python3 work/build_manuscript.py` from the included `workspace/` directory. All required text/style inputs are included and listed in `source_inventory.json`; no dataset, inference, or terminal LaTeX installation is needed to build the source. Open the resulting standalone `.tex` in the native editor to compile. The source is a working research revision, not a submitted paper or certification of conference readiness.
