# Manuscript source

`revised_manuscript.tex` is the self-contained manuscript. It embeds the AISTATS style and all three vector figures, so it does not require external figure files. The style credits and source URL are retained.

To assemble the same source from its components:

```sh
python paper/build.py --output /tmp/revised_manuscript.tex
```

This command assembles LaTeX source; it does not compile or export a PDF. The `source/` directory contains the main narrative, experimental section, selected appendix, and figure fragments. Scientific figure/table generation is documented in the repository root.
