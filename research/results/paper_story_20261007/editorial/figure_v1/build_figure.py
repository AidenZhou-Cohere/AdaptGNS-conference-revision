"""Restyle the existing Goop paired figure; no new results or scientific runs."""
from pathlib import Path
import hashlib
import json
import math
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'work/conference_presentation_20261006/goop_integration_candidate_v1/figure_coordinates_and_source_AUDIT_PENDING.json'
SOURCE_SHA = '3e2289459c065f6bfebd1f8055f4f5c863c595431ada0f17ff06b125c0a332b4'
WIDTH, HEIGHT = 470, 264
COLORS = {'ink': '#263444', 'muted': '#62717F', 'grid': '#E4E9ED',
          'baseline': '#A5AFB8', 'seed0': '#0072B2', 'seed1': '#B56520',
          'seed2': '#268577', 'white': '#FFFFFF'}
CAPTION = (
    r'\textbf{Training on added edges improves Goop rollouts, but risk still trails random placement.} '
    r'Three paired seeds are trained from initialization for 100,000 updates. '
    r'\textbf{A,} random25 position-coordinate MSE falls in every seed '
    r'(30 complete H395 rollouts per seed and arm). '
    r'\textbf{B,} risk minus random remains positive at 150 matched observed histories per model, '
    r'although every gap narrows; risk uses the preceding observed base graph. '
    r'Lines pair seeds; black bars mark three-seed means. \textbf{Endpoints and scales differ.} '
    r'Across the full study, 1,077/1,080 rollouts complete; mixed seed-2 guards affect base, dense '
    r'and cached risk. The mixed cached-risk full-horizon mean and its training interaction remain undefined.'
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    expected = SOURCE_SHA
    assert sha(SOURCE) == expected
    source = json.loads(SOURCE.read_bytes())
    records = source['records']
    primitives, points, means = [], [], []

    def text(x, y, plain, latex=None, size=8, align='left', bold=False, color='ink'):
        primitives.append(dict(kind='text', x=x, y=y, plain=plain,
                               latex=plain if latex is None else latex,
                               size=size, align=align, bold=bold, color=color))

    def line(x1, y1, x2, y2, color='ink', width=.7, dashed=False):
        primitives.append(dict(kind='line', x1=x1, y1=y1, x2=x2, y2=y2,
                               color=color, width=width, dashed=dashed))

    def marker(x, y, seed):
        primitives.append(dict(kind='marker', x=x, y=y, seed=seed, color=f'seed{seed}'))

    text(17, 249, 'A', size=11.5, bold=True)
    text(34, 249, 'Exposure helps rollouts', size=11.5, bold=True)
    text(255, 249, 'B', size=11.5, bold=True)
    text(272, 249, 'Risk still trails random', size=11.5, bold=True)
    text(17, 232, 'Random25 · all 395 forecasts', r'Random25 $\cdot$ all 395 forecasts', size=8.5)
    text(255, 232, '150 matched observed histories', size=8.5)
    text(17, 217, 'Position-coordinate MSE', size=7.7, color='muted')
    text(255, 217, 'Risk − random MSE (×10⁻¹⁰)', r'Risk $-$ random MSE ($\times10^{-10}$)', size=7.7, color='muted')

    y0, y1 = 73, 203
    panels = (
        ('exposure', 43, 218, .55, (0, .1, .2, .3, .4, .5),
         (84, 182), ('random_base', 'random_mix'), 1.),
        ('placement', 281, 456, 9., (0, 3, 6, 9),
         (322, 420), ('gap_base', 'gap_mix'), 1e10),
    )
    for panel, left, right, top, ticks, xs, names, scale in panels:
        for tick in ticks:
            y = y0 + tick / top * (y1-y0)
            line(left, y, right, y, color='baseline' if tick == 0 else 'grid',
                 width=.7 if tick == 0 else .35)
            text(left-8, y, f'{tick:.1f}' if panel == 'exposure' else str(tick),
                 size=7.5, align='right', color='muted')
        for seed in range(3):
            values = [records[name]['seed_values'][str(seed)] for name in names]
            ys = [y0 + value*scale/top*(y1-y0) for value in values]
            sx = [x+(seed-1)*5 for x in xs]
            line(sx[0], ys[0], sx[1], ys[1], color=f'seed{seed}', width=1.05,
                 dashed=seed == 2)
            for name, raw, x, y in zip(names, values, sx, ys):
                marker(x, y, seed)
                points.append(dict(panel=panel, record=name, seed=seed, raw_value=raw,
                                   display_value=raw*scale, x=x, y=y, scale=scale))
        for x, name in zip(xs, names):
            raw=records[name]['mean'];y=y0+raw*scale/top*(y1-y0)
            line(x-11, y, x+11, y, width=2.2)
            means.append(dict(panel=panel, record=name, raw_value=raw,
                              display_value=raw*scale, x=x, y=y, scale=scale))
        for x, label in zip(xs, ('Base-only', 'Mixed-graph')):
            text(x, 58, label, size=8, align='center', bold=True)
            text(x, 46, 'training', size=7.5, align='center', color='muted')

    text(130, 28, 'Mean  0.301 → 0.129', r'Mean\quad $0.301\rightarrow0.129$',
         size=8.5, align='center', bold=True)
    text(368, 28, 'Mean gap  +6.39 → +2.84', r'Mean gap\quad $+6.39\rightarrow+2.84$',
         size=8.5, align='center', bold=True)
    # Shared key keeps colors assigned to paired training seeds across both panels.
    for x, seed in ((95, 0), (160, 1), (225, 2)):
        marker(x, 9, seed)
        text(x+9, 9, f'Seed {seed}', size=7.5, color='muted')
    line(290, 9, 307, 9, width=2.2)
    text(315, 9, 'Three-seed mean', size=7.5, color='muted')

    def solid(x1, y1, x2, y2):
        return rf'\qbezier({x1:.9f},{y1:.9f})({(x1+x2)/2:.9f},{(y1+y2)/2:.9f})({x2:.9f},{y2:.9f})'

    picture=[r'\begingroup', r'\sffamily', r'\setlength{\unitlength}{1pt}']
    for name, value in COLORS.items():
        picture.append(rf'\definecolor{{goop{name}}}{{HTML}}{{{value[1:]}}}')
    picture.append(rf'\begin{{picture}}({WIDTH},{HEIGHT})')
    for p in primitives:
        picture.append(r'\color{goop'+p['color']+'}')
        if p['kind'] == 'line':
            picture.append(rf'\linethickness{{{p["width"]}pt}}')
            if p['dashed']:
                length=math.hypot(p['x2']-p['x1'],p['y2']-p['y1'])
                count=max(1,math.ceil(length/6))
                for i in range(count):
                    a=i/count;b=min(1,(i+.64)/count)
                    picture.append(solid(p['x1']+(p['x2']-p['x1'])*a,p['y1']+(p['y2']-p['y1'])*a,
                                         p['x1']+(p['x2']-p['x1'])*b,p['y1']+(p['y2']-p['y1'])*b))
            else:picture.append(solid(p['x1'],p['y1'],p['x2'],p['y2']))
        elif p['kind'] == 'marker':
            x,y,seed=p['x'],p['y'],p['seed']
            picture.append(rf'\put({x:.9f},{y:.9f}){{\color{{goopwhite}}\circle*{{6.2}}}}')
            picture.append(rf'\linethickness{{0.9pt}}')
            if seed in (0,1):
                symbol=r'\circle{5.2}' if seed==0 else r'\circle*{5.2}'
                picture.append(rf'\put({x:.9f},{y:.9f}){{{symbol}}}')
            else:
                coords=((x,y+3.0),(x-2.8,y-2.3),(x+2.8,y-2.3))
                for a,b in zip(coords,coords[1:]+coords[:1]):picture.append(solid(*a,*b))
        else:
            pos={'left':'[l]','right':'[r]','center':''}[p['align']]
            font=rf'\fontsize{{{p["size"]}}}{{{p["size"]+1}}}\selectfont'
            if p['bold']:font+=r'\bfseries'
            picture.append(rf'\put({p["x"]:.9f},{p["y"]:.9f})'+'{'+r'\makebox(0,0)'+pos+'{'+font+' '+p['latex']+'}}')
    picture.extend([r'\end{picture}',r'\endgroup',''])
    picture_text='\n'.join(picture)
    (HERE/'goop_first_results_picture.tex').write_text(picture_text)
    fragment='\n'.join([r'\begin{figure*}[t]',r'\centering',picture_text,
                         r'\caption{'+CAPTION+'}',r'\label{fig:goop-exposure-placement}',r'\end{figure*}',''])
    (HERE/'goop_first_results_figure.tex').write_text(fragment)
    (HERE/'preview_standalone.tex').write_text('\n'.join([
        r'\documentclass[border=6pt]{standalone}',r'\usepackage{xcolor}',r'\begin{document}',
        picture_text,r'\end{document}','']))

    os.environ.setdefault('MPLCONFIGDIR',str(HERE/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    fig=plt.figure(figsize=(WIDTH/72,HEIGHT/72),dpi=240)
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,WIDTH);ax.set_ylim(0,HEIGHT);ax.axis('off')
    texts=[]
    for p in primitives:
        color=COLORS[p['color']]
        if p['kind']=='line':
            ax.plot([p['x1'],p['x2']],[p['y1'],p['y2']],color=color,lw=p['width'],
                    linestyle=(0,(4,2.25)) if p['dashed'] else '-',solid_capstyle='butt',zorder=1)
        elif p['kind']=='marker':
            ax.plot(p['x'],p['y'],marker=('o','o','^')[p['seed']],markersize=5.2,
                    markerfacecolor=color if p['seed']==1 else 'white',markeredgecolor=color,
                    markeredgewidth=.9,linestyle='none',zorder=3)
        else:
            texts.append(ax.text(p['x'],p['y'],p['plain'],fontsize=p['size'],ha=p['align'],
                                 va='center',fontweight='bold' if p['bold'] else 'normal',color=color))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    text_boxes=[list(x.get_window_extent(renderer).bounds) for x in texts]
    # Preview-only text bounds check; native font metrics remain root's compile check.
    canvas=fig.bbox
    assert all(x>=0 and y>=0 and x+w<=canvas.width+.5 and y+h<=canvas.height+.5 for x,y,w,h in text_boxes)
    for ext in ('svg','png'):fig.savefig(HERE/f'goop_first_results.{ext}',dpi=240)
    plt.close(fig)

    old={(p['record'],p['seed']):p for p in source['plotted_values']}
    assert len(points)==len(old)==12 and len(means)==4
    for point in points:
        match=old[(point['record'],point['seed'])]
        assert point['raw_value']==match['raw_value']
        assert point['display_value']==match['display_value']
    for mean in means:assert mean['raw_value']==records[mean['record']]['mean']
    assert records['cached_risk_full_horizon_mix']['mean'] is None
    assert records['cached_risk_full_horizon_interaction']['mean'] is None
    assert source['coverage']=={'completed_required_outcome':1077,'recorded_failed_outcome':3}
    assert sha(SOURCE)==expected
    assert r'\includegraphics' not in fragment and r'\begin{tikzpicture}' not in fragment
    record={'schema':'goop_first_results_presentation_v1','source':str(SOURCE),'source_sha256':expected,
            'source_summary_sha256':source['summary_sha256'],'source_records_unchanged':records,
            'dimensions_pt':[WIDTH,HEIGHT],'points':points,'means':means,'primitives':primitives,
            'caption_latex':CAPTION,'fidelity':{'exact_points':12,'exact_means':4,
            'endpoints_and_scales_explicit':True,'three_seed_pairing_preserved':True,
            'full_cohort_guard_and_undefined_cached_risk_retained':True,
            'native_compilation_performed':False,'matplotlib_preview_bounds_passed':True,
            'scientific_runs_or_array_reads':False,'canonical_files_edited':False},
            'products_sha256':{name:sha(HERE/name) for name in ('goop_first_results_picture.tex',
             'goop_first_results_figure.tex','preview_standalone.tex','goop_first_results.svg','goop_first_results.png')}}
    (HERE/'figure_fidelity.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'points':len(points),'means':len(means),'figure_sha256':sha(HERE/'goop_first_results_figure.tex'),
                      'fidelity_sha256':sha(HERE/'figure_fidelity.json')},sort_keys=True))


if __name__=='__main__':main()
