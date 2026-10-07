from pathlib import Path
import re
import argparse
ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Assemble the standalone manuscript source; no TeX compiler is invoked.")
parser.add_argument("--output", type=Path, default=ROOT / "revised_manuscript.tex")
args = parser.parse_args()
style=(ROOT / 'source/aistats2027.sty').read_text()
style=re.sub(r'\\NeedsTeXFormat[^\n]*\n','',style)
style=re.sub(r'\\ProvidesPackage[^\n]*\n','',style)
start=style.index(r'\DeclareOption{accepted}')
end=style.index(r'\ProcessOptions\relax')+len(r'\ProcessOptions\relax')
style=style[:start]+r'''
% Embedded default submission layout; no external project files required.
\newcommand{\statePaper}{0}
\newcommand{\acceptedPaper}{1}
\newcommand{\Notice@String}{}
\newcommand{\AISTATS@appearing}{}
\newif\if@preprint
\@preprintfalse
'''+style[end:]
body=(ROOT / 'source/manuscript_body.tex').read_text()
# Assemble the manuscript and its self-contained figure sources.
for marker, name in (
    ('% CONFERENCE_EXPERIMENTS_MAIN_INSERT', 'conference_experiments_main.tex'),
    ('% CURATED_APPENDIX_INSERT', 'curated_appendix.tex'),
    ('% VISUAL_OVERVIEW_INSERT', 'visual_overview_figure.tex'),
    ('% VISUAL_EVIDENCE_INSERT', 'visual_evidence_figure.tex'),
    ('% VISUAL_UNCERTAINTY_TIME_INSERT', 'visual_uncertainty_times.tex'),
    ('% VISUAL_UNCERTAINTY_ASSOCIATIONS_INSERT', 'visual_uncertainty_associations.tex'),
    ('% VISUAL_PARTICLE_INSERT', 'visual_particle_figure.tex')):
    insert = ROOT / 'source' / name
    body = body.replace(marker, insert.read_text())
body = body.replace('% GOOP3D_AUTONOMOUS_H295_SEPARATE_INSERT', '')
text=r'''\documentclass[twoside,letterpaper]{article}
% Official AISTATS 2027 layout embedded for the single-file native editor.
% Source: https://aistats.org/aistats2027/AISTATS2027PaperPack.zip
% Package option handling is adapted; the venue notice is suppressed.
\makeatletter
'''+style+r'''
\makeatother
\usepackage{amssymb,amsthm,booktabs,xcolor}
\usepackage[round]{natbib}
\usepackage[hidelinks]{hyperref}
\newtheorem{proposition}{Proposition}
\newcommand{\sg}{\operatorname{sg}}
'''+body
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(text)
print(args.output)
