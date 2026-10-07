"""Re-layout already verified particle glyphs; no array or model reads."""
from pathlib import Path
import hashlib, json, re, os

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(os.environ.get('REPRODUCTION_OUTPUT', ROOT/'generated'))
HERE.mkdir(parents=True,exist_ok=True)
SOURCE = ROOT/'results/goop2d/qualitative_glyphs.json'
data = json.loads(SOURCE.read_text())
panels = [0,8,9,10,11]
labels = [('Truth','Forecast 1'), ('Truth','Forecast 395'), ('Base training','Base graph'), ('Mixed training','Random25'), ('Mixed training','Cached risk25')]
colors = ['truth','truth','base','random','risk']
palette = {'ink':'263444','muted':'667783','grid':'C8D2D9','truth':'247E87','base':'697B8C','random':'207C63','risk':'A55A31','outside':'B74442'}
W,H=470,187
commands = [r'\begingroup\sffamily\setlength{\unitlength}{1pt}']
for k,v in palette.items(): commands.append(r'\definecolor{pv'+k+r'}{HTML}{'+v+'}')
commands.append(r'\begin{picture}(470,187)')
texts=[]; lines=[]; plotted=[]
def text(x,y,s,size=8,bold=False,color='ink',latex=None):
    texts.append((x,y,s,size,bold,color))
    font=rf'\fontsize{{{size}}}{{{size+1}}}\selectfont'+(r'\bfseries' if bold else '')
    commands.append(rf'\put({x},{y}){{\color{{pv{color}}}\makebox(0,0){{{font} '+(s if latex is None else latex)+'}}')
def line(x1,y1,x2,y2):
    lines.append((x1,y1,x2,y2))
    commands.append(rf'\color{{pvgrid}}\linethickness{{0.4pt}}\qbezier({x1},{y1})({(x1+x2)/2},{(y1+y2)/2})({x2},{y2})')
text(235,176,'Full-horizon forecasts can remain visibly wrong',11,True)
text(286,158,'Four views of the same final forecast',8,color='muted')
for column,index in enumerate(panels):
    oldrow,oldcol=divmod(index,4);cx=48+93.5*column;bottom=44;size=82
    oldcx=66+116*oldcol;oldcy=302-116*oldrow
    selected=data['panels'][str(index)]
    points=[(cx+(x-oldcx)*size/83,bottom+size/2+(y-oldcy)*size/83,dot) for x,y,dot in selected]
    assert len(points)==1083
    outside=sum(not dot for _,_,dot in points)
    assert outside==[0,0,465,11,14][column]
    text(cx,144,labels[column][0],8,True)
    text(cx,133,labels[column][1],7.7,color='muted')
    for a,b in (((cx-size/2,bottom),(cx+size/2,bottom)),((cx-size/2,bottom+size),(cx+size/2,bottom+size)),((cx-size/2,bottom),(cx-size/2,bottom+size)),((cx+size/2,bottom),(cx+size/2,bottom+size))):line(*a,*b)
    commands.append(r'\color{pv'+colors[column]+'}')
    for x,y,dot in points:
        if dot:commands.append(rf'\put({x:.6f},{y:.6f}){{\circle*{{0.7}}}}')
        else:commands.append(rf'\put({x:.6f},{y:.6f}){{\color{{pvoutside}}\makebox(0,0){{\fontsize{{4.5}}{{5}}\selectfont$\times$}}}}')
    text(cx,31,f'Outside: {outside}/1,083',7.5,color='muted')
    plotted.append({'original_panel':index,'glyphs':points,'outside':outside})
text(235,10,'Same spatial bounds and all particles; crosses mark clamped out-of-box positions.',7.2,color='muted')
commands.extend([r'\end{picture}',r'\endgroup'])
picture='\n'.join(commands)+'\n'
caption=(r'\textbf{Full-horizon forecasts can remain visibly wrong.} '
 r'A fixed Goop example (source 12, seed 0) selected by median initial particle count, without inspecting errors. '
 r'All 1,083 particles use the same bounds at forecasts 1 and 395. The target settles along the floor; predicted groups remain suspended or leave the box. '
 r'Training and policy both differ across prediction columns, so this is an illustration of physical mismatch, not a paired training-effect estimate. '
 r'Crosses show any strict box crossing, clamped to the boundary; they do not encode excursion magnitude.')
(HERE/'qualitative_strip_picture.tex').write_text(picture)
(HERE/'qualitative_strip_figure.tex').write_text(r'\begin{figure*}[t]'+'\n'+r'\centering'+'\n'+picture+r'\caption{'+caption+'}\n'+r'\label{fig:qualitative-goop2d}'+'\n'+r'\end{figure*}'+'\n')
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='DejaVu Sans'
plt.rcParams['svg.hashsalt']='adaptive-interaction-graphs'
fig=plt.figure(figsize=(W/72,H/72),dpi=240);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off')
for x1,y1,x2,y2 in lines:ax.plot([x1,x2],[y1,y2],color='#'+palette['grid'],lw=.4)
for k,p in enumerate(plotted):
    dots=[(x,y) for x,y,d in p['glyphs'] if d];cross=[(x,y) for x,y,d in p['glyphs'] if not d]
    ax.scatter(*zip(*dots),s=.7**2,c='#'+palette[colors[k]],linewidths=0)
    if cross:ax.scatter(*zip(*cross),s=2.2**2,c='#'+palette['outside'],marker='x',linewidths=.3)
for x,y,s,size,bold,color in texts:ax.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',ha='center',va='center',color='#'+palette[color])
for ext in ('png','svg'):fig.savefig(HERE/f'qualitative_strip.{ext}',dpi=240,metadata={'Date':None} if ext=='svg' else None)
plt.close(fig)
(HERE/'qualitative_counts.json').write_text(json.dumps({'panels':panels,'particles_each':1083,'outside_counts':[p['outside'] for p in plotted]},indent=2)+'\n')
print('Qualitative figure: five panels, 5,415 saved glyphs')
