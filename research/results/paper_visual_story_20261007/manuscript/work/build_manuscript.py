from pathlib import Path
import re
style=Path('work/sources/aistats2027.sty').read_text()
style=re.sub(r'\\NeedsTeXFormat[^\n]*\n','',style)
style=re.sub(r'\\ProvidesPackage[^\n]*\n','',style)
start=style.index(r'\DeclareOption{accepted}')
end=style.index(r'\ProcessOptions\relax')+len(r'\ProcessOptions\relax')
style=style[:start]+r'''
% Embedded default submission layout; no external project files required.
\newcommand{\statePaper}{0}
\newcommand{\acceptedPaper}{1}
\newcommand{\Notice@String}{Working research revision. October 4, 2026.}
\newcommand{\AISTATS@appearing}{}
\newif\if@preprint
\@preprintfalse
'''+style[end:]
body=Path('work/manuscript_body.tex').read_text()
# Only current paper fragments enter the standalone editor document.
# Historical material stays in the repository and cannot reappear implicitly.
for marker, name in (
    ('% CONFERENCE_EXPERIMENTS_MAIN_INSERT', 'conference_experiments_main.tex'),
    ('% CURATED_APPENDIX_INSERT', 'curated_appendix.tex'),
    ('% VISUAL_OVERVIEW_INSERT', 'visual_overview_figure.tex'),
    ('% VISUAL_EVIDENCE_INSERT', 'visual_evidence_figure.tex'),
    ('% VISUAL_PARTICLE_INSERT', 'visual_particle_figure.tex'),
    ('% GOOP3D_AUTONOMOUS_H295_SEPARATE_INSERT', 'goop3d_autonomous_appendix.tex')):
    insert = Path('work') / name
    body = body.replace(marker, insert.read_text() if insert.exists() else '')
text=r'''\documentclass[twoside,letterpaper]{article}
% Official AISTATS 2027 layout embedded for the single-file native editor.
% Source: https://aistats.org/aistats2027/AISTATS2027PaperPack.zip
% Only package option handling and working-draft notice are adapted.
\makeatletter
'''+style+r'''
\makeatother
\usepackage{amssymb,amsthm,booktabs,xcolor}
\usepackage[round]{natbib}
\usepackage[hidelinks]{hyperref}
\newtheorem{proposition}{Proposition}
\newcommand{\sg}{\operatorname{sg}}
'''+body
Path('outputs/revised_manuscript.tex').write_text(text)
