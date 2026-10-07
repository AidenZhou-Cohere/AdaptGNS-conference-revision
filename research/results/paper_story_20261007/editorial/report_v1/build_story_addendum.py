from pathlib import Path
import hashlib,json,shutil
from pypdf import PdfReader,PdfWriter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
R=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
report=R/'outputs/revision_report.pdf'
assert hashlib.sha256(report.read_bytes()).hexdigest()=='bb76664ad0aea0989438af9ad7474f1fd0d109aa6482816e255ef9a741149cbb'
source=R/'work/goop3d_completion_20261007/report_addendum_recipe_v1/displayed_observed_statistics.json'
stats=json.loads(source.read_text());assert len(stats)==18
fig=R/'work/paper_story_20261007/figure_v1/goop_first_results.png'
before=PdfReader(report);assert len(before.pages)==21
backup=HERE/'revision_report_before.pdf';assert not backup.exists();shutil.copy2(report,backup)
ink=colors.HexColor('#263444');muted=colors.HexColor('#62717F');blue=colors.HexColor('#0072B2')
styles={
 'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=20,leading=24,textColor=ink,spaceAfter=7),
 'deck':ParagraphStyle('deck',fontName='Helvetica',fontSize=9,leading=12,textColor=muted,spaceAfter=12),
 'sub':ParagraphStyle('sub',fontName='Helvetica-Bold',fontSize=11,leading=14,textColor=blue,spaceBefore=10,spaceAfter=4),
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=7,textColor=ink),
 'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8.5,leading=11,textColor=muted,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=8.5,leading=11,textColor=ink),
}
story=[];prose=[]
def para(t,k='body'):
 assert t.isascii(),t;story.append(Paragraph(t,styles[k]));prose.append({'style':k,'text':t})
def table(rows,widths):
 t=Table([[Paragraph(str(c),styles['cell']) for c in row] for row in rows],colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF0F4')),('LINEBELOW',(0,0),(-1,0),.7,muted),('LINEBELOW',(0,-1),(-1,-1),.5,muted),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t);story.append(Spacer(1,8))
para('The story: learning to use extra messages','title')
para('Current research synthesis | October 7, 2026 | Author-facing companion to the revised paper','deck')
para('An adaptive particle simulator faces two learning problems: using additional interactions and choosing which interactions will help. A cached residual controller makes the second decision cheaply, while paired graph-exposure training tests the first. The experiments show why the two cannot be treated as the same problem.')
im=Image(str(fig),width=510,height=510*880/1566);story.append(im);story.append(Spacer(1,6))
para('Goop, three paired training seeds. Left: the same Random25 policy over 395 forecasts, 30 trajectories per seed and arm. Right: risk minus random on 150 matched observed histories; risk uses the preceding observed base graph. The panels have different endpoints and scales.','small')
para('1  Extra messages become more useful after exposure','sub')
para('Goop Random25 rollout MSE falls from 0.301 to 0.129 after mixed-graph training, improving every seed. WaterDrop paired continuations also improve every seed at this fixed policy. These comparisons measure a training effect; they do not establish that expansion beats the native graph.')
para('2  Prediction error is an imperfect guide to action','sub')
para('Goop risk still loses to random on matched histories. WaterDrop reverses that test ordering after exposure, but not on average on validation. Sand narrows its observed risk deficit while still losing to random. The useful target is the benefit of a graph change, which residual error alone does not identify.')
para('3  Feedback is the decisive test','sub')
para('Sand cached-risk rollout error worsens in every seed after exposure, despite its narrower observed deficit. Native base beats random expansion in both arms. WaterDrop has mixed autonomous interactions; three Goop guards leave the affected full-horizon effects undefined. Substantial boundary excursions remain, and measured timing does not establish an isolated speedup.')
para('The completed evidence supports a controlled construction and a design lesson: train for the graphs the model will encounter, evaluate placement on common histories, and test the resulting feedback. The Goop3D observed extension is summarized next; its autonomous evaluation is still in progress.','small')
story.append(PageBreak())
para('Goop3D: does the distinction persist?','title')
para('Completed observed-history extension | Six 25k endpoints | Three paired seeds | Autonomous results pending','deck')
para('The 3D extension asks whether exposure and placement remain distinct at a shorter training endpoint. All 2,568 observed cells are complete: 768 clean-validation frames plus 900 same-state cells on each split. No autonomous Goop3D accuracy conclusion is included here.')
policies=['base','dense','random25','speed25','previous-observed-base-risk25','relative-velocity-RMS25']
labels=['Base','Dense','Random25','Speed25','Previous risk25','RMS25'];displayed={}
def take(k):
 v=stats[k]['statistic'];assert set(v['seed_values'])=={'0','1','2'} and v['mean'] is not None;displayed[k]=stats[k];return v
def fmt(v):return f"{v['mean']*1e9:+.3f} +/- {v['sample_sd']*1e9:.3f}"
def signs(v):return '/'.join('+' if v['seed_values'][str(i)]>0 else '-' if v['seed_values'][str(i)]<0 else '0' for i in range(3))
para('Training changes do not have a uniform sign','sub')
rows=[['Evaluation policy','Validation: mix - base<br/>Mean +/- SD [seed signs]','Test: mix - base<br/>Mean +/- SD [seed signs]']]
for p,l in zip(policies,labels):
 a=take('same_state_valid/mix_minus_base/'+p);b=take('same_state_test/mix_minus_base/'+p);rows.append([l,fmt(a)+' ['+signs(a)+']',fmt(b)+' ['+signs(b)+']'])
table(rows,[122,201,201])
para('Position-coordinate MSE x 10<super>-9</super>. Negative favors mixed training. Signs retain seeds 0/1/2. Random25 exposure improves two validation seeds but only one test seed, despite a lower mean on both splits. The 25k endpoint is distinct from the 100k/110k studies.','small')
para('The test placement gap narrows, but remains positive','sub')
rows=[['Risk - random contrast','Validation mean +/- SD','Test mean +/- SD']]
for stem,label in [('risk_minus_random/base','Base-only training'),('risk_minus_random/mix','Mixed-graph training'),('risk_interaction','Change in gap')]:
 a=take('same_state_valid/'+stem);b=take('same_state_test/'+stem);rows.append([label,fmt(a),fmt(b)])
table(rows,[180,172,172])
para('On test, exposure narrows the gap in every seed, but risk remains worse than random in two mixed-arm seeds and on average. Validation differs: risk beats random in two seeds of each arm, with a mixed-sign training interaction. A smaller gap is therefore not equivalent to consistently better placement.')
para('Observed risk uses scores from the preceding observed base graph. Autonomous cached risk uses its own selected graph and predicted history. These findings cannot establish the autonomous ordering. Means and sample SDs describe three paired seeds; SD is not a confidence interval.','small')
url='https://github.com/AidenZhou-Cohere/AdaptGNS-conference-revision/blob/323ff7d8c4de15a41d70cdeb3b3c9fdaae1d0593/research/results/goop3d_completion_20261007/observed_products/summary.json'
para(f'<link href="{url}">Complete observed-policy companion: every policy, split and seed.</link> The preceding 21 report pages retain their historical content. Final autonomous results and the submission PDF remain separate work.','small')
def footer(c,d):
 c.setFont('Helvetica',8);c.setFillColor(muted);c.drawString(44,24,'Adaptive Interaction Graphs | current scientific synthesis');c.drawRightString(568,24,str(21+d.page))
add=HERE/'story_addendum.pdf';candidate=HERE/'revision_report_candidate.pdf';assert not add.exists() and not candidate.exists()
SimpleDocTemplate(str(add),pagesize=(612,792),leftMargin=44,rightMargin=44,topMargin=36,bottomMargin=43).build(story,onFirstPage=footer,onLaterPages=footer)
a=PdfReader(add);assert len(a.pages)==2,len(a.pages)
w=PdfWriter();w.append(PdfReader(backup));w.append(a);w.add_outline_item('Current synthesis: the story',21);w.add_outline_item('Goop3D observed-history extension',22)
w.add_metadata({'/Title':'Adaptive Interaction Graphs - research companion','/Subject':'Completed findings and current reader-facing synthesis','/Author':'Author-facing research revision'})
with candidate.open('xb') as f:w.write(f)
c=PdfReader(candidate);assert len(c.pages)==23
checks=[]
for i in range(21):
 x,y=before.pages[i],c.pages[i];assert x.get_contents().get_data()==y.get_contents().get_data();assert x.extract_text()==y.extract_text();assert x.mediabox==y.mediabox;assert x.get('/Rotate',0)==y.get('/Rotate',0);checks.append(i+1)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(report)==sha(backup)
receipt={'status':'structure_passed_pending_visual_review','source_report_sha256':sha(report),'candidate_sha256':sha(candidate),'preserved_original_pages':checks,'new_pages':2,'inputs':{str(x.relative_to(R)):sha(x) for x in [source,fig]},'new_source_sha256':sha(Path(__file__)),'all18_prepared_observed_objects_retained':len(displayed)==18,'no_autonomous_goop3d_result':True}
(HERE/'report_structure_review.json').write_text(json.dumps(receipt,indent=2)+'\n');(HERE/'displayed_observed_statistics.json').write_text(json.dumps(displayed,indent=2)+'\n');(HERE/'prose.json').write_text(json.dumps(prose,indent=2)+'\n')
print(json.dumps(receipt))
