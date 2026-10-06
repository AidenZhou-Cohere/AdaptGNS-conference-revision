"""Append two pages from already admitted scalar evidence; no scientific reruns."""
from pathlib import Path
import hashlib, gzip, json, math, statistics, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Image, Table, TableStyle, PageBreak, SimpleDocTemplate
W=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;R=W/'outputs/AdaptGNS/research/results'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gp=R/'goop2d_graph_exposure_100k_20261006/paired_scalar_summary.json.gz';wp=R/'waterdrop_continuation_110k_20261006/summary/result.json'
assert sha(gp)=='bb3c3f54cdeaf2b1198f49b8437dc711d80ce5e8347ab70363558ab515852978'
assert sha(wp)=='082bdc9c33614fea6445e3373e94eeafcfdfc516d4a27d04096f380ad7781e64'
g=json.loads(gzip.decompress(gp.read_bytes()));w=json.loads(wp.read_text());f=g['full_rollout'];wm=w['populations']['autonomous_test']['metrics']['mean_rollout_mse']
assert f['coverage']=={'completed_required_outcome':1077,'recorded_failed_outcome':3}
assert w['coverage']=={'endpoints':6,'jobs':12,'observed_frames':2550,'observed_policy_slots':12750,'autonomous_outcomes':810}
assert all(j['failed_records']==0 for j in w['jobs'])
report=W/'outputs/revision_report.pdf';assert sha(report)=='f412f1f809c96f1dbd0f75844be2fbda11dc6f8f0dc0020bbb1bea1839440574'
backup=D/'revision_report_before.pdf'
if not backup.exists():shutil.copyfile(report,backup)
assert sha(backup)==sha(report) and len(PdfReader(backup).pages)==18
checks=[];data={}
def vals(v):return [v['seed_values'][str(s)] for s in range(3)] if isinstance(v['seed_values'],dict) else v['seed_values']
def checked(v,key):
 x=vals(v);assert len(x)==3
 if any(z is None for z in x):assert v['mean'] is None and v['sample_sd'] is None
 else:
  assert math.isclose(statistics.mean(x),v['mean'],rel_tol=1e-10,abs_tol=1e-14)
  assert math.isclose(statistics.stdev(x),v['sample_sd'],rel_tol=1e-10,abs_tol=1e-14)
 checks.append(key);data[key]=v;return v
labels={'base':'Base','dense':'Dense','random25':'Random25','speed25':'Speed25','relative-velocity-RMS25':'Relative-velocity RMS25','laggedrisk25':'Cached risk25'}
policies=list(labels)
gabs=f['absolute']['mean_rollout_mse'];gdelta=f['mix_minus_base']['mean_rollout_mse'];wabs=wm['absolute'];wdelta=wm['paired']
def fmt(v,n):return 'undefined' if v['mean'] is None else f"{v['mean']:.{n}f} +/- {v['sample_sd']:.{n}f}"
styles={
 'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=18,leading=22,textColor=colors.HexColor('#16334a'),spaceAfter=9),
 'sub':ParagraphStyle('sub',fontName='Helvetica-Bold',fontSize=11,leading=15,textColor=colors.HexColor('#16334a'),spaceBefore=6,spaceAfter=5),
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8.5,leading=11,textColor=colors.HexColor('#455767'),spaceAfter=6),
 'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=8,leading=11)}
story=[]
def para(text,style='body'):story.append(Paragraph(text,styles[style]))
def table(rows):
 t=Table([[Paragraph(str(x),styles['cell']) for x in row] for row in rows],colWidths=[132,133,133,126],repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eff3')),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#708291')),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#708291')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('BACKGROUND',(0,3),(-1,3),colors.HexColor('#f0f7f5'))]))
 story.extend([t,Spacer(1,8)])
def plot(absolute,name,horizon):
 a=vals(checked(absolute['base']['random25'],name+'/random/base'));b=vals(checked(absolute['mix']['random25'],name+'/random/mix'));assert all(y<x for x,y in zip(a,b))
 fig,ax=plt.subplots(figsize=(7.4,2.1));palette=['#147d92','#b0652d','#7d5ba6']
 for seed,(x,y) in enumerate(zip(a,b)):
  ax.plot([0,1],[x,y],marker='o',color=palette[seed],lw=1.6,ms=5,label=f'Seed {seed}')
 ax.set_xticks([0,1],['Base-only training','Mixed-graph training']);ax.set_xlim(-.25,1.35);ax.set_ylim(0,max(a+b)*1.12);ax.set_ylabel(f'H{horizon} position MSE');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.legend(loc='upper right',frameon=False,fontsize=8);fig.tight_layout();p=D/(name+'_paired_random.png');fig.savefig(p,dpi=180);plt.close(fig);story.extend([Image(str(p),width=510,height=145),Spacer(1,3)])
para('17. Graph exposure: fresh Goop training','head')
para('October 6 scientific update. The preceding 18 pages retain their earlier dated snapshot. This addendum supersedes their in-progress descriptions of Goop2D100k and WaterDrop110k.','small')
para('Six faithful models train from initialization to 100k updates on all 1,000 official training trajectories. Paired arms share initialization, sampled frames and noise. Mixed training expands half of examples with a uniform quarter of optional annulus pairs; native base messages remain intact.')
plot(gabs,'goop2d',395)
para('<b>Fixed-policy improvement.</b> Random25 full-rollout MSE improves in every paired seed. Lines connect the same seed; each value averages 30 trajectory means. The graph-exposure intervention is separate from choosing where to place extra messages.','small')
rows=[['Inference policy','Base-only: mean +/- SD','Mixed: mean +/- SD','Mix minus base: mean +/- SD']]
for p in policies:
 a=checked(gabs['base'][p],'goop/'+p+'/base');b=checked(gabs['mix'][p],'goop/'+p+'/mix');d=checked(gdelta[p],'goop/'+p+'/delta')
 for x,y,z in zip(vals(a),vals(b),vals(d)):assert z is None if x is None or y is None else math.isclose(y-x,z,rel_tol=1e-10,abs_tol=1e-14)
 rows.append([labels[p],fmt(a,3),fmt(b,3),fmt(d,3)])
table(rows)
para('Coverage and interpretation','sub')
para('All 1,080 planned H395 outcomes are retained: 1,077 complete and three candidate-pair guards, one each for mixed seed 2 base, dense and cached risk. Their full three-seed means and the risk-versus-random training interaction remain undefined. No survivor mean replaces them.','small')
para('On matched observed test histories, risk still loses to random in every mixed-model seed. Mixed speed and relative-velocity RMS beat random in every full-rollout seed. Thus exposure helps this simulator, while residual ranking is not established as a superior placement rule.','small')
out=f['absolute']['predicted_boundary_mean_fraction_particles_outside_by_more_than_1e-6'];truth=f['absolute']['ground_truth_boundary_mean_fraction_particles_outside_by_more_than_1e-6']['base']['random25']['mean']
para(f"Mixed random, speed and RMS retain {100*out['mix']['random25']['mean']:.2f}%, {100*out['mix']['speed25']['mean']:.2f}% and {100*out['mix']['relative-velocity-RMS25']['mean']:.2f}% of particles outside the metadata box by more than 1e-6, versus {100*truth:.4f}% for truth. Box excursions do not test conservation. Shared-host, policy-dependent rollout timing does not establish isolated speedup.",'small')
para('Source: goop2d_graph_exposure_100k_20261006/paired_scalar_summary.json.gz and its admitted saved-array audits. Checks cover saved positions and scalar series; unsaved trajectories and source/model execution are not replayed. Three-seed means and sample SDs are descriptive.','small')
story.append(PageBreak())
para('18. Graph exposure: WaterDrop continuation','head')
para('Three fixed faithful 100k parents each receive paired 10k base-only and mixed-graph continuations, inheriting optimizer state and matched frame/noise schedules. The endpoint is 110k; this is conditional continuation evidence, not fresh 110k training.','body')
plot(wabs,'waterdrop110k',995)
para('<b>Fixed-policy improvement.</b> Random25 improves in every paired seed, as do dense, speed and cached risk. Each seed equally averages 27 trajectory means. Different material scales and training budgets prevent an absolute-error comparison with Goop.','small')
rows=[['Inference policy','Base-only: mean +/- SD','Mixed: mean +/- SD','Mix minus base: mean +/- SD']]
for p in [p for p in policies if p!='relative-velocity-RMS25']:
 a=checked(wabs['base'][p],'waterdrop/'+p+'/base');b=checked(wabs['mix'][p],'waterdrop/'+p+'/mix');d=checked(wdelta['mix_minus_base__'+p],'waterdrop/'+p+'/delta')
 for x,y,z in zip(vals(a),vals(b),vals(d)):assert math.isclose(y-x,z,rel_tol=1e-10,abs_tol=1e-14)
 rows.append([labels[p],fmt(a,4),fmt(b,4),fmt(d,4)])
table(rows)
para('Placement and physical behavior','sub')
para('All 810 H995 outcomes complete with zero scientific failures. All 2,550 observed histories and 12,750 policy slots are retained. Observed-test risk changes from worse than random to better in every seed after exposure. Validation does not share a universal reversal: mixed risk remains worse than random on average.','small')
v=checked(wdelta['risk_minus_random_interaction'],'waterdrop/autonomous_risk_interaction');assert min(vals(v))<0<max(vals(v))
para(f"The autonomous risk-minus-random training interaction is {v['mean']:+.5f} +/- {v['sample_sd']:.5f}, with mixed seed signs. Observed-state scores are computed on a preceding observed base history; autonomous cached scores follow the policy's own graph. An observed-state gain therefore does not establish a consistent autonomous placement benefit.",'small')
para('Mixed random, speed and cached-risk rollouts still place more than 16% of particles outside the metadata box, versus 2.68% for truth. Recorded times and edge counts increase for every policy after exposure. These fixed-order, geometry-dependent costs remain descriptive. Numerical completion does not certify physical validity.','small')
para('Scientific takeaway','sub')
para('Across these two exploratory studies, training on expanded graphs improves full rollouts under an unchanged random policy in every paired seed. Useful placement is a separate question, with adverse or mixed risk results. The evidence supports graph exposure as a constructive intervention; it does not establish universal adaptive-allocation or efficiency superiority.','small')
para('Source: waterdrop_continuation_110k_20261006/summary/result.json, its 91 metric-detail files and independent audit. All earlier fixed-study evidence remains unchanged. This addendum includes no interim Sand or Goop 3D accuracy claims.','small')
def footer(canvas,doc):
 canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#617383'));canvas.drawString(44,25,'Adaptive Interaction Graphs | completed graph-exposure studies');canvas.drawRightString(568,25,str(18+doc.page))
add=D/'graph_exposure_addendum.pdf';SimpleDocTemplate(str(add),pagesize=(612,792),leftMargin=44,rightMargin=44,topMargin=38,bottomMargin=43).build(story,onFirstPage=footer,onLaterPages=footer)
assert len(PdfReader(add).pages)==2, 'Keep addendum concise and legible in two pages'
writer=PdfWriter();writer.append(PdfReader(backup));writer.append(PdfReader(add));candidate=D/'revision_report_candidate.pdf'
with candidate.open('wb') as out:writer.write(out)
assert len(PdfReader(candidate).pages)==20
(D/'report_checks.json').write_text(json.dumps({'status':'scalar_extraction_and_pdf_structure_passed_pending_visual_review','source_sha256':{str(gp.relative_to(W)):sha(gp),str(wp.relative_to(W)):sha(wp)},'original_pdf_sha256':sha(backup),'candidate_pdf_sha256':sha(candidate),'addendum_pdf_sha256':sha(add),'original_pages':18,'candidate_pages':20,'statistic_objects_checked':len(checks),'checks':checks,'scientific_runs_or_original_audits_reexecuted':False},indent=2)+'\n')
(D/'displayed_statistic_objects.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({'candidate':str(candidate),'pages':20,'checks':len(checks),'sha256':sha(candidate)}))
