"""Render measured WaterDrop message counts and all published observed policy rows."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from picture import Picture, COLORS

DATA_SHA = 'a054ab4d130f8e04e9013d57f1cb7f5ecff68108f397a06f8f63656e7f1c008f'
MAP_SHA = '6b51a4de77719f5036e32ae79bddd280876deed0d648084b198b48f3e15a00b5'
POLICIES = ('base', 'random25', 'speed25', 'previous-observed-base-risk25', 'relative-velocity-RMS25', 'dense')
COLORS.update(purple='#866196')
STYLE = {'base': ('N', 'ink'), 'dense': ('D', 'native'), 'random25': ('R', 'teal'),
         'speed25': ('S', 'blue'), 'previous-observed-base-risk25': ('Q', 'rust'),
         'relative-velocity-RMS25': ('V', 'purple')}


class ScopedPicture(Picture):
    """Keep each native primitive's color local, including inside every put."""
    def native(self):
        out = [r'\begingroup', r'\sffamily', r'\setlength{\unitlength}{1pt}']
        for k, v in COLORS.items():
            out.append(r'\definecolor{pvs' + k + '}{HTML}{' + v[1:] + '}')
        out.append(r'\begin{picture}(' + str(self.width) + ',' + str(self.height) + ')')
        for p in self.items:
            c = r'\color{pvs' + p['color'] + '}'
            if p['kind'] == 'line':
                count = max(1, math.ceil(math.hypot(p['x2']-p['x1'], p['y2']-p['y1']) / 5)) if p['dashed'] else 1
                for i in range(count):
                    a, b = i/count, min(1, (i + (.58 if p['dashed'] else 1))/count)
                    x1, y1 = p['x1'] + (p['x2']-p['x1'])*a, p['y1'] + (p['y2']-p['y1'])*a
                    x2, y2 = p['x1'] + (p['x2']-p['x1'])*b, p['y1'] + (p['y2']-p['y1'])*b
                    out.append('{' + c + r'\linethickness{' + str(p['width']) + 'pt}' +
                               rf'\qbezier({x1:.9f},{y1:.9f})({(x1+x2)/2:.9f},{(y1+y2)/2:.9f})({x2:.9f},{y2:.9f})' + '}')
            elif p['kind'] == 'circle':
                put = rf'\put({p["x"]:.9f},{p["y"]:.9f})'
                if not p['filled']:
                    out.append(put + r'{\color{pvswhite}\circle*{' + str(2*p['r']) + '}}')
                out.append(put + '{' + c + r'\linethickness{' + str(p['width']) + 'pt}' +
                           r'\circle' + ('*' if p['filled'] else '') + '{' + str(2*p['r']) + '}}')
            elif p['kind'] == 'rect':
                out.append(rf'\put({p["x"]:.9f},{p["y"]:.9f})' + '{' + c +
                           r'\rule{' + str(p['w']) + 'pt}{' + str(p['h']) + 'pt}}')
            else:
                align = {'left':'[l]', 'right':'[r]', 'center':''}[p['align']]
                font = r'\fontsize{' + str(p['size']) + '}{' + str(p['size']+1) + r'}\selectfont' + (r'\bfseries' if p['bold'] else r'\mdseries')
                out.append(rf'\put({p["x"]:.9f},{p["y"]:.9f})' + '{' + c + r'\makebox(0,0)' + align + '{' + font + ' ' + p['tex'] + '}}')
        return '\n'.join(out + [r'\end{picture}', r'\endgroup', ''])


def ordered(stat):
    values = stat['seed_values']
    return [values[str(i)] for i in range(3)] if isinstance(values, dict) else values


def symbol(p, x, y, seed, color, r):
    if seed == 0:
        p.circle(x, y, r=r, color=color)
    elif seed == 1:
        p.circle(x, y, r=r, color=color, filled=False, width=.65)
    else:
        p.cross(x, y, r=r, color=color, width=.7)


def draw(data, output):
    assert data['rows_required'] == 34 and data['seed_values_required'] == 102
    assert data['source_scalar_map_sha256'] == MAP_SHA
    assert data['autonomous_data_included'] is False
    rows = {(r['material'], r['training_arm'], r['policy']): r for r in data['rows']}
    edges = {(r['training_arm'], r['policy']): r for r in data['waterdrop_total_edges_context']}
    assert len(rows) == 34 and len(edges) == 10
    assert all(r['ordered_seed_values'] == ordered(r['statistic_unchanged']) for r in rows.values())
    assert all(v is not None and math.isfinite(v) for r in rows.values() for v in r['ordered_seed_values'])
    p = ScopedPicture(470, 300)
    points, means = [], []
    p.text(8, 310, 'WaterDrop: accuracy versus total messages', size=12, bold=True)
    p.text(8, 296, 'Observed-test coordinate MSE ×10⁻⁹', r'Observed-test coordinate MSE $\times10^{-9}$', size=7.6, color='muted')
    for x, text, policy in ((227,'N native','base'),(273,'R random','random25'),(326,'S speed','speed25'),(374,'Q risk','previous-observed-base-risk25'),(419,'D dense','dense')):
        p.text(x, 296, text, size=7.0, color=STYLE[policy][1], bold=True)
    sparse_offsets = {'base':0, 'dense':0, 'random25':-8, 'speed25':0, 'previous-observed-base-risk25':8}
    for arm, left in (('base', 34), ('mix', 272)):
        width, bottom, top = 188, 172, 274
        x = lambda value: left + (value-4450)/(7650-4450)*width
        y = lambda value: bottom + (value*1e9-3.2)/(4.8-3.2)*(top-bottom)
        p.text(left+width/2, 285, 'Base-only training' if arm=='base' else 'Mixed training', size=9.2, bold=True, align='center')
        for tick in (3.2, 3.6, 4.0, 4.4, 4.8):
            yy = y(tick*1e-9)
            p.line(left, yy, left+width, yy, color='grid', width=.4)
            p.text(left-5, yy, f'{tick:.1f}', size=7.2, align='right', color='muted')
        for policy, label in (('base','4,802'), ('random25','5,418'), ('dense','7,269')):
            xx = x(edges[arm,policy]['statistic_unchanged']['mean'])
            p.line(xx, bottom, xx, top, color='grid', width=.55)
            p.line(xx, bottom-3, xx, bottom, color='zero', width=.65)
            p.text(xx, 161, label, size=7.5, align='center', color='muted')
        p.line(left, bottom, left+width, bottom, color='zero', width=.7)
        p.line(left, bottom, left, top, color='zero', width=.7)
        native = rows['WaterDrop',arm,'base']['statistic_unchanged']['mean']
        p.line(left,y(native),left+width,y(native),color='zero',width=.75,dashed=True)
        for policy in POLICIES:
            if policy == 'relative-velocity-RMS25':
                continue
            row, edge = rows['WaterDrop',arm,policy], edges[arm,policy]
            color = STYLE[policy][1]
            edge_values = ordered(edge['statistic_unchanged'])
            for seed, value in enumerate(row['ordered_seed_values']):
                raw_x = edge_values[seed]
                xx = x(raw_x) + sparse_offsets[policy] + (seed-1)*2.3
                yy = y(value)
                symbol(p,xx,yy,seed,color,1.6)
                points.append({'material':'WaterDrop','training_arm':arm,'policy':policy,'seed':seed,
                    'raw_mse':value,'display_multiplier':1e9,'raw_mean_total_directed_edges':raw_x,
                    'exact_x_before_readability_offset':x(raw_x),'horizontal_readability_offset_pt':xx-x(raw_x),
                    'x':xx,'y':yy,'source_key':row['source_key'],'edge_source_key':edge['source_key']})
            mean, mean_x = row['statistic_unchanged']['mean'], edge['statistic_unchanged']['mean']
            p.line(x(mean_x)-4.5,y(mean),x(mean_x)+4.5,y(mean),color=color,width=1.5)
            means.append({'material':'WaterDrop','training_arm':arm,'policy':policy,'raw_mse':mean,
                          'raw_mean_total_directed_edges':mean_x,'x':x(mean_x),'y':y(mean),'source_key':row['source_key'],
                          'edge_source_key':edge['source_key']})
    p.text(235,147,'Mean total directed messages; mean bars at exact x, seed symbols offset',size=7.5,align='center',color='muted')
    p.line(8,139,462,139,color='grid',width=.6)
    p.text(112,130,'Goop · optional-message budgets',r'Goop: optional-message budgets',size=8.6,bold=True,align='center')
    p.text(350,130,'Sand · optional-message budgets',r'Sand: optional-message budgets',size=8.6,bold=True,align='center')
    for material, arm, left in (('Goop','base',28),('Goop','mix',141),('Sand','base',266),('Sand','mix',379)):
        width,bottom,top=80,49,106
        scale = 1e8
        ylim,ticks = ((1.1,1.65),(1.2,1.4,1.6)) if material=='Goop' else ((2.7,7.5),(3,5,7))
        y = lambda value: bottom+(value*scale-ylim[0])/(ylim[1]-ylim[0])*(top-bottom)
        p.text(left+width/2,118,'Base-only' if arm=='base' else 'Mixed',size=7.4,bold=True,align='center')
        for tick in ticks:
            yy=y(tick/scale)
            p.line(left,yy,left+width,yy,color='grid',width=.35)
            p.text(left-4,yy,f'{tick:g}',size=6.5,align='right',color='muted')
        p.line(left,bottom,left,top,color='zero',width=.6)
        native=rows[material,arm,'base']['statistic_unchanged']['mean']
        p.line(left,y(native),left+width,y(native),color='zero',width=.6,dashed=True)
        xs = {policy:left+6+i*(width-12)/5 for i,policy in enumerate(POLICIES)}
        for policy in POLICIES:
            row,color = rows[material,arm,policy],STYLE[policy][1]
            for seed,value in enumerate(row['ordered_seed_values']):
                xx,yy=xs[policy]+(seed-1)*1.9,y(value)
                symbol(p,xx,yy,seed,color,1.1)
                points.append({'material':material,'training_arm':arm,'policy':policy,'seed':seed,
                    'raw_mse':value,'display_multiplier':scale,'x':xx,'y':yy,'source_key':row['source_key'],
                    'nominal_optional_budget_fraction':row['nominal_optional_budget_fraction'],
                    'axis_type':'categorical optional-message budget; not measured total edges'})
            mean=row['statistic_unchanged']['mean']
            p.line(xs[policy]-3.5,y(mean),xs[policy]+3.5,y(mean),color=color,width=1.1)
            means.append({'material':material,'training_arm':arm,'policy':policy,'raw_mse':mean,'source_key':row['source_key']})
            p.text(xs[policy],40,STYLE[policy][0],size=7.0,bold=True,align='center',color=color)
        lo,hi=xs['random25']-4,xs['relative-velocity-RMS25']+4
        p.line(lo,33,hi,33,color='zero',width=.55)
        p.line(lo,33,lo,36,color='zero',width=.55)
        p.line(hi,33,hi,36,color='zero',width=.55)
        p.text(xs['base'],29,'0',size=6.6,align='center',color='muted')
        p.text((lo+hi)/2,26,'≤25%',r'$\le25\%$',size=6.6,align='center',color='muted')
        p.text(xs['dense'],29,'100%',r'$100\%$',size=6.6,align='center',color='muted')
    p.text(235,18,'Lower panels: coordinate MSE ×10⁻⁸; V = relative-velocity RMS',r'Lower panels: coordinate MSE $\times10^{-8}$; V = relative-velocity RMS',size=6.9,color='muted',align='center')
    symbol(p,17,6,0,'ink',1.4);p.text(22,6,'seed 0',size=6.8)
    symbol(p,71,6,1,'ink',1.4);p.text(76,6,'seed 1',size=6.8)
    symbol(p,125,6,2,'ink',1.4);p.text(130,6,'seed 2',size=6.8)
    p.line(178,6,188,6,width=1.25);p.text(193,6,'mean',size=6.8)
    p.line(239,6,258,6,color='zero',dashed=True);p.text(264,6,'native mean',size=6.8,color='muted')
    p.text(462,6,'Lower error is better',size=6.8,color='muted',align='right')
    caption = (r'\textbf{Selective expansion can improve accuracy at a smaller message budget.} '
        r'Within each training arm, policies use the same fixed models and observed test histories. A separately dense-trained model is not included. '
        r'WaterDrop (top) plots position-coordinate MSE against mean total directed messages; counts are identical across arms. '
        r'Mean bars use exact counts; seed symbols are offset horizontally for legibility. '
        r'Goop/Sand (bottom) retain categorical fractions of dense optional annulus additions, with native messages retained; their axes do not show measured total counts. '
        r'Q uses preceding observed-base risk. Symbols show three seeds per policy and arm; dashed lines mark native means. '
        r'These observed-state comparisons do not establish runtime or autonomous gains.')
    assert len(points)==102 and len(means)==34
    # Compress only layout coordinates to the 300pt footprint; data and type sizes are unchanged.
    for item in p.items:
        for key in ('y', 'y1', 'y2', 'h'):
            if key in item:
                item[key] *= 300 / 320
    for item in points + means:
        if 'y' in item:
            item['y'] *= 300 / 320
    product=p.save(output,'observed_accuracy_budget',caption,'fig:observed-accuracy-budget')
    figure_data={'schema':'observed_accuracy_measured_waterdrop_budget_figure_v2','display_data_sha256':DATA_SHA,
       'source_scalar_map_sha256':MAP_SHA,'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'picture_helper_sha256':hashlib.sha256((Path(__file__).parent/'picture.py').read_bytes()).hexdigest(),
       'dimensions_pt':[470,300],'seed_points':points,'mean_marks':means,'source_row_count':34,'expected_seed_count':102,
       'all_declared_policies_retained':True,'autonomous_data_included':False,
       'waterdrop_mean_bars_at_exact_empirical_x':True,'seed_horizontal_offsets_are_visual_only':True,'product_sha256':product['sha256']}
    (output/'observed_accuracy_budget_data.json').write_text(json.dumps(figure_data,indent=2,sort_keys=True)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    here=Path(__file__).resolve().parent
    packaged=here.parent/'results/cross_material/budget_display_data.json'
    parser.add_argument('--data',type=Path,default=packaged if packaged.is_file() else here/'display_data.json')
    default_output = here.parent/'generated' if packaged.is_file() else here/'generated'
    parser.add_argument('--output-dir',type=Path,default=Path(os.environ.get('REPRODUCTION_OUTPUT',default_output)))
    args=parser.parse_args()
    raw=args.data.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==DATA_SHA
    args.output_dir.mkdir(parents=True,exist_ok=True)
    draw(json.loads(raw),args.output_dir)


if __name__=='__main__':
    main()
