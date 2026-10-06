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
pilot=Path('work/pilot_results.tex')
if pilot.exists():body=body.replace('% PILOT_RESULTS_INSERT',pilot.read_text())
else:body=body.replace('% PILOT_RESULTS_INSERT',r'\noindent\textit{The planned pilot is running; numerical results have not yet been inserted into this working copy.}')
rollout=Path('work/rollout_results.tex')
if rollout.exists():body=body.replace('% ROLLOUT_RESULTS_INSERT',rollout.read_text())
for marker, name in (('% CONFERENCE_EXPERIMENTS_MAIN_INSERT', 'conference_experiments_main.tex'),
                     ('% FULL_EVALUATION_MAIN_INSERT', 'full_evaluation_main.tex'),
                     ('% FULL_EVALUATION_APPENDIX_INSERT', 'full_evaluation_appendix.tex'),
                     ('% FULL_ACTION_MAIN_INSERT', 'full_action_main.tex'),
                     ('% FULL_ACTION_APPENDIX_INSERT', 'full_action_appendix.tex'),
                     ('% GRAPH_BRIDGE_MAIN_INSERT', 'graph_bridge_main.tex'),
                     ('% GRAPH_BRIDGE_APPENDIX_INSERT', 'graph_bridge_appendix.tex'),
                     ('% NATIVE_FOLLOWUP_APPENDIX_INSERT', 'native_followup_appendix.tex'),
                     ('% OPTIONAL_EXPOSURE_MAIN_INSERT', 'optional_exposure_main.tex'),
                     ('% OPTIONAL_EXPOSURE_APPENDIX_INSERT', 'optional_exposure_appendix.tex'),
                     ('% NOISE_AUGMENTATION_INSERT', 'noise_augmentation_appendix.tex'),
                     ('% TIE_SYMMETRY_INSERT', 'tie_symmetry_appendix.tex'),
                     ('% NONADDITIVE_ALLOCATION_INSERT', 'nonadditive_allocation_appendix.tex')):
    insert = Path('work') / name
    body = body.replace(marker, insert.read_text() if insert.exists() else '')
text=r'''\documentclass[twoside,letterpaper]{article}
% Official AISTATS 2027 layout embedded for the single-file native editor.
% Source: https://aistats.org/aistats2027/AISTATS2027PaperPack.zip
% Only package option handling and working-draft notice are adapted.
\makeatletter
'''+style+r'''
\makeatother
\usepackage{amssymb,amsthm,booktabs}
\usepackage[round]{natbib}
\usepackage[hidelinks]{hyperref}
\newtheorem{proposition}{Proposition}
\newcommand{\sg}{\operatorname{sg}}
'''+body
Path('outputs/revised_manuscript.tex').write_text(text)
