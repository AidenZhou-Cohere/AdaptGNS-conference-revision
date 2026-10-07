# First results figure

`goop_first_results_figure.tex` is a self-contained, double-column LaTeX figure ready to place near the introduction. It preserves `fig:goop-exposure-placement`. Replace the old figure with this fragment; do not retain both copies of the label. The source uses only native `picture` primitives, standard LaTeX fonts, and `xcolor` (`\usepackage{xcolor}` in the manuscript preamble). It does not include external images or require TikZ/PGF.

`goop_first_results_picture.tex` contains only the grouped picture, and `preview_standalone.tex` wraps the same picture for an optional standalone preview. Native compilation is left to root. The PNG and SVG were rendered from the same primitive/coordinate list as the LaTeX fragment; they show intended geometry and hierarchy, while native font metrics remain a compile-time check.

The figure leads with the result in each panel, keeps seed colors and symbols consistent, distinguishes individual seed pairs from the dark mean bars, and gives the endpoints separate scales and explicit subtitles. Zero remains visible in both panels. The caption preserves the full-cohort guard caveat and the undefined cached-risk full-horizon comparison. The guard affects policies outside the complete random25 endpoint in panel A and is not hidden or substituted away.

Rebuild locally with:

```sh
work/venv/bin/python -B work/paper_story_20261007/figure_v1/build_figure.py
```

The only numerical input is the existing source-pinned figure-coordinate/record JSON. `figure_fidelity.json` records all 12 exact seed values, four exact means, endpoint scales, new coordinates, original source records (including nulls), and output hashes. No array/model/result generation, new experiment, canonical manuscript edit, or native compile was performed.

Suggested placement: use this as Figure 1 immediately after the introduction establishes exposure versus placement. Let the paper's opening picture answer its scientific question. A general graph-construction schematic can then follow the method where its symbols and mechanics have context.
