"""Shared native-LaTeX and matplotlib drawing primitives, with no science IO."""
from pathlib import Path
import hashlib
import json
import math
import os

COLORS={'ink':'#263544','muted':'#607281','native':'#768A98','grid':'#DFE6EA',
        'wash':'#F1F5F6','teal':'#197F79','rust':'#B76432','blue':'#3A7BA5',
        'white':'#FFFFFF','zero':'#93A3AE'}


class Picture:
    def __init__(self,width,height):self.width,self.height,self.items=width,height,[]
    def text(self,x,y,plain,tex=None,size=8,align='left',bold=False,color='ink'):
        self.items.append(dict(kind='text',x=x,y=y,plain=plain,tex=plain if tex is None else tex,size=size,align=align,bold=bold,color=color))
    def line(self,x1,y1,x2,y2,color='ink',width=.7,dashed=False):
        self.items.append(dict(kind='line',x1=x1,y1=y1,x2=x2,y2=y2,color=color,width=width,dashed=dashed))
    def circle(self,x,y,r=2,color='ink',filled=True,width=.8):
        self.items.append(dict(kind='circle',x=x,y=y,r=r,color=color,filled=filled,width=width))
    def rect(self,x,y,w,h,color='wash'):
        self.items.append(dict(kind='rect',x=x,y=y,w=w,h=h,color=color))
    def cross(self,x,y,r=2,color='muted',width=.9):
        self.line(x-r,y-r,x+r,y+r,color,width);self.line(x-r,y+r,x+r,y-r,color,width)
    def diamond(self,x,y,r=2.5,color='ink'):
        self.circle(x,y,r+.8,'white')
        corners=((x-r,y),(x,y+r),(x+r,y),(x,y-r))
        for a,b in zip(corners,corners[1:]+corners[:1]):self.line(*a,*b,color=color,width=.9)
    def arrow(self,x1,y1,x2,y2,color='muted',width=.8,head=3):
        self.line(x1,y1,x2,y2,color,width)
        angle=math.atan2(y2-y1,x2-x1)
        for sign in (-1,1):
            a=angle+sign*.45;self.line(x2-head*math.cos(a),y2-head*math.sin(a),x2,y2,color,width)

    def native(self):
        out=[r'\begingroup',r'\sffamily',r'\setlength{\unitlength}{1pt}']
        for k,v in COLORS.items():out.append(r'\definecolor{pvs'+k+'}{HTML}{'+v[1:]+'}')
        out.append(r'\begin{picture}('+str(self.width)+','+str(self.height)+')')
        def line(x1,y1,x2,y2):
            return rf'\qbezier({x1:.9f},{y1:.9f})({(x1+x2)/2:.9f},{(y1+y2)/2:.9f})({x2:.9f},{y2:.9f})'
        for p in self.items:
            out.append(r'\color{pvs'+p['color']+'}')
            if p['kind']=='line':
                out.append(r'\linethickness{'+str(p['width'])+'pt}')
                count=max(1,math.ceil(math.hypot(p['x2']-p['x1'],p['y2']-p['y1'])/5)) if p['dashed'] else 1
                for i in range(count):
                    a=i/count;b=min(1,(i+(.58 if p['dashed'] else 1))/count)
                    out.append(line(p['x1']+(p['x2']-p['x1'])*a,p['y1']+(p['y2']-p['y1'])*a,p['x1']+(p['x2']-p['x1'])*b,p['y1']+(p['y2']-p['y1'])*b))
            elif p['kind']=='circle':
                out.append(r'\linethickness{'+str(p['width'])+'pt}')
                if not p['filled']:out.append(rf'\put({p["x"]:.9f},{p["y"]:.9f})'+'{'+r'\color{pvswhite}\circle*{'+str(2*p['r'])+'}}')
                out.append(rf'\put({p["x"]:.9f},{p["y"]:.9f})'+'{'+r'\circle'+('*' if p['filled'] else '')+'{'+str(2*p['r'])+'}}')
            elif p['kind']=='rect':
                out.append(rf'\put({p["x"]:.9f},{p["y"]:.9f})'+'{'+r'\rule{'+str(p['w'])+'pt}{'+str(p['h'])+'pt}}')
            else:
                align={'left':'[l]','right':'[r]','center':''}[p['align']]
                font=r'\fontsize{'+str(p['size'])+'}{'+str(p['size']+1)+'}'+r'\selectfont'+(r'\bfseries' if p['bold'] else '')
                out.append(rf'\put({p["x"]:.9f},{p["y"]:.9f})'+'{'+r'\makebox(0,0)'+align+'{'+font+' '+p['tex']+'}}')
        return '\n'.join(out+[r'\end{picture}',r'\endgroup',''])

    def save(self,where,name,caption,label):
        where=Path(where);native=self.native()
        (where/(name+'_picture.tex')).write_text(native)
        (where/(name+'.tex')).write_text('\n'.join([r'\begin{figure*}[t]',r'\centering',native,r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{figure*}','']))
        os.environ.setdefault('MPLCONFIGDIR',str(where/'matplotlib_cache'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle,Rectangle
        plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
        fig=plt.figure(figsize=(self.width/72,self.height/72),dpi=240);ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,self.width);ax.set_ylim(0,self.height);ax.axis('off');texts=[]
        for z,p in enumerate(self.items):
            color=COLORS[p['color']]
            if p['kind']=='line':ax.plot([p['x1'],p['x2']],[p['y1'],p['y2']],color=color,lw=p['width'],linestyle=(0,(3,2.2)) if p['dashed'] else '-',solid_capstyle='butt',zorder=z)
            elif p['kind']=='circle':ax.add_patch(Circle((p['x'],p['y']),p['r'],facecolor=color if p['filled'] else 'white',edgecolor=color,lw=0 if p['filled'] else p['width'],zorder=z))
            elif p['kind']=='rect':ax.add_patch(Rectangle((p['x'],p['y']),p['w'],p['h'],facecolor=color,edgecolor='none',zorder=z))
            else:texts.append(ax.text(p['x'],p['y'],p['plain'],color=color,fontsize=p['size'],ha=p['align'],va='center',fontweight='bold' if p['bold'] else 'normal',zorder=z))
        fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[]
        for text in texts:
            box=text.get_window_extent(renderer);bounds.append({'text':text.get_text(),'bbox':list(box.bounds)})
            assert box.x0>=-.5 and box.y0>=-.5 and box.x1<=fig.bbox.width+.5 and box.y1<=fig.bbox.height+.5,(text.get_text(),box.bounds)
        for ext in ('png','svg'):fig.savefig(where/(name+'.'+ext),dpi=240)
        plt.close(fig)
        product={'dimensions_pt':[self.width,self.height],'primitives':self.items,'caption_latex':caption,'label':label,'preview_text_bounds':bounds,
                 'sha256':{name+ext:hashlib.sha256((where/(name+ext)).read_bytes()).hexdigest() for ext in ('.tex','_picture.tex','.png','.svg')}}
        (where/(name+'_primitives.json')).write_text(json.dumps(product,sort_keys=True,indent=2)+'\n')
        return product
