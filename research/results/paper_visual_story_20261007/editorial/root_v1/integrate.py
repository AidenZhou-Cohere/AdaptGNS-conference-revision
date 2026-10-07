"""Integrate the reviewed visual manuscript into the existing editor source."""
from pathlib import Path
import hashlib,json,re,shutil
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
old=(HERE/'before/manuscript_body.tex').read_text()
main=(HERE/'main_reviewed.tex').read_text()
for i,name in ((1,'VISUAL_OVERVIEW_INSERT'),(2,'VISUAL_EVIDENCE_INSERT'),(3,'VISUAL_PARTICLE_INSERT')):
    main,n=re.subn(r'% FIGURE '+str(i)+r':[^\n]*','% '+name,main,count=1)
    assert n==1
start=main.index(r'\section{Experiments}')
end=main.index(r'\section{From residual risk to interaction benefit}')
experiments=main[start:end]
main=main[:start]+'% CONFERENCE_EXPERIMENTS_MAIN_INSERT\n\n'+main[end:]
header=old[:old.index(r'\begin{abstract}')]
back=old[old.index('\\clearpage\n\\typeout{REVISION-MAIN-PAGES:'):old.index(r'\appendix')]
ai_start=back.index(r'\section*{AI Use Statement}')
ai_end=back.index(r'\begin{thebibliography}',ai_start)
back=back[:ai_start]+r'''\section*{AI Use Statement}
OpenAI Codex assisted with literature retrieval, implementation, experiment execution, analysis and writing. \textit{Draft disclosure: the author must finalize the scope of human verification and responsibility before submission.}

'''+back[ai_end:]
answers={
'The revision gives graph-cardinality and cost decompositions and limited timing measurements, but not a complete time/space complexity analysis for every evaluated implementation.':'Graph cardinality, selection complexity and measured costs are provided; a complete complexity analysis for every implementation is not.',
'Tested source, dependency specifications and compact result records cover the new studies. A complete anonymized submission package and its accessible release have not yet been verified.':'Code and dependencies are preserved; the anonymous submission package remains to be finalized.',
'Proofs are supplied for the propositions and additional derivations. This describes their inclusion, not certification of independent human verification.':'Proofs and derivations are included in the supplementary material.',
'Scripts, fixed protocols and measurement artifacts cover the historical-array reanalysis, compact pilot, fixed full-model study and completed exploratory controls. A complete anonymized reproduction package has not yet been verified; the original historical checkpoints and complete historical training/evaluation provenance remain unavailable.':'Scripts, protocols and result records are available in the research repository. A complete anonymous reproduction package remains to be finalized.',
'The compact-pilot, benefit-head and fixed full-model protocols specify their data splits, architectures, objectives, noise, hyperparameters and checkpoint-selection rules. The exploratory controls have explicit input and evaluation protocols. Exact historical training and validation-selection details remain unverified.':'Training protocols, data splits and fixed endpoints are described; final author verification of the complete reproduction instructions remains pending.',
'Metrics distinguish normalized acceleration from position error, trajectory-conditioned bootstrap intervals from training-seed standard deviations, and failed from completed rollouts.':'Captions define position error, paired seed effects, sample standard deviations and undefined full-horizon outcomes.',
'The local CPU/Metal setup, thread counts, relevant versions, timing scope, and device-parity limitations are recorded with the experiment artifacts.':'Per-study devices, software versions, timing scope and backend limitations are recorded with the experiment artifacts.',
'Software license notices are retained in the code package, but complete asset-license information has not yet been consolidated into the submission.':'Software notices are retained; complete asset-license information remains to be finalized.',
'New source, fitted coefficients and compact measurement records are preserved alongside local checkpoints and numeric archives. A complete anonymized submission archive or accessible release has not yet been verified.':'Code, checkpoints and measurements are preserved. The anonymous submission archive remains to be finalized.'}
for a,b in answers.items():
    assert a in back,a
    back=back.replace(a,b)
body=header+main+'\n'+back+'% CURATED_APPENDIX_INSERT\n\n\\end{document}\n'
copies={
 'visual_overview_figure.tex':ROOT/'work/paper_visual_story_20261007/figures_v1/constructive_overview.tex',
 'visual_evidence_figure.tex':ROOT/'work/paper_visual_story_20261007/figures_v1/cross_material_evidence.tex',
 'visual_particle_figure.tex':HERE/'qualitative_strip/qualitative_strip_figure.tex',
 'curated_appendix.tex':ROOT/'work/paper_visual_story_20261007/appendix_review/curated_appendix.tex'}
for name,p in copies.items():shutil.copyfile(p,ROOT/'work'/name)
(ROOT/'work/manuscript_body.tex').write_text(body)
(ROOT/'work/conference_experiments_main.tex').write_text(experiments)
builder=(HERE/'before/build_manuscript.py').read_text()
a=builder.index("pilot=Path('work/pilot_results.tex')")
b=builder.index("text=r'''",a)
builder=builder[:a]+'''# Only current paper fragments enter the standalone editor document.
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
'''+builder[b:]
(ROOT/'work/build_manuscript.py').write_text(builder)
record={'before_source_sha256':hashlib.sha256((HERE/'before/revised_manuscript.tex').read_bytes()).hexdigest(),
 'integrated_inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'work/manuscript_body.tex',ROOT/'work/conference_experiments_main.tex',ROOT/'work/build_manuscript.py',*[ROOT/'work'/n for n in copies]]},
 'scope':'Editorial restructuring in same open file. No scientific rerun. Old inputs and publications retained.'}
(HERE/'integration_inputs.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'canonical_body_bytes':len(body.encode()),'experiment_fragment_bytes':len(experiments.encode()),'fragments':list(copies)}))
