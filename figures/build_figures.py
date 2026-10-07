"""Render the two quantitative paper figures from saved seed statistics."""
from pathlib import Path
import hashlib
import json
from picture import Picture

import os
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(os.environ.get('REPRODUCTION_OUTPUT', ROOT/'generated'))
HERE.mkdir(parents=True,exist_ok=True)
OLD=ROOT/'results/cross_material'
PINS={'printed_statistic_map.json':'76fe8163513e82166d4f5e4eda0e400c7f74fed68949162f8202287043cf041f',
      'all_scalar_statistic_map.json':'6b51a4de77719f5036e32ae79bddd280876deed0d648084b198b48f3e15a00b5'}
MATERIALS=('Goop','WaterDrop','Sand')
OBS=(('mix_minus_base_at_base','Train effect: native'),('mix_minus_base_at_random25','Train effect: random'),
     ('risk_minus_random_base_training','Risk gap: base'),('risk_minus_random_mixed_training','Risk gap: mixed'),
     ('risk_minus_random_training_interaction','Change in risk gap'))
POLICIES=(('base','Base'),('dense','Dense'),('random25','Random25'),('speed25','Speed25'),
          ('laggedrisk25','Cached risk25'),('relative-velocity-RMS25','RMS25'))


def values(record):
    x=record['seed_values']
    return [x[str(i)] for i in range(3)] if isinstance(x,dict) else list(x)


def load():
    docs={}
    for name,pin in PINS.items():
        raw=(OLD/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==pin;docs[name]=json.loads(raw)
    data=docs['printed_statistic_map.json'];all_stats=docs['all_scalar_statistic_map.json']
    feedback={}
    for m in MATERIALS:
        key=('populations/autonomous_test/metrics/mean_rollout_mse/paired/risk_minus_random_interaction' if m=='WaterDrop'
             else 'full_rollout/risk_minus_random_mix_minus_base_interaction/mean_rollout_mse')
        feedback[m]={'source_key':key,'record':all_stats[m][key]}
    return data,feedback


def overview(data,feedback):
    p=Picture(470,238)
    p.text(8,224,'Preserve accuracy. Spend fewer extra interactions.',size=12.5,bold=True)
    coords=((11,38),(18,54),(34,33),(38,50),(53,61),(60,43),(22,76),(44,79))
    native=((0,1),(2,3),(3,4),(4,5))
    optional=((0,2),(1,3),(1,6),(3,5),(4,7),(6,7),(1,2),(5,7))
    for stage,(left,title) in enumerate(((35,'Native graph'),(197,'Dense expansion'),(359,'Quarter-budget expansion'))):
        center=left+25
        p.text(center,202,title,size=8.5,align='center',bold=True)
        nodes=[(left+(x-11),139+(y-33)*.95) for x,y in coords]
        for a,b in native:p.line(*nodes[a],*nodes[b],color='native',width=1.1)
        if stage==1:
            for a,b in optional:p.line(*nodes[a],*nodes[b],color='blue',width=.8)
        if stage==2:
            p.line(*nodes[0],*nodes[2],color='teal',width=2.2)
            p.line(*nodes[4],*nodes[7],color='teal',width=2.2)
            p.circle(*nodes[2],r=4.3,color='teal',filled=False,width=1)
        for x,y in nodes:p.circle(x,y,r=2.1,color='ink')
        p.text(center,129,('4 native pairs','4 native + 8 added pairs','4 native + 2 added pairs')[stage],
               (None,None,None)[stage],size=8,align='center',color='muted')
    p.arrow(103,162,177,162,color='zero',width=1.1,head=5)
    p.arrow(265,162,339,162,color='zero',width=1.1,head=5)
    p.text(140,176,'add every candidate',size=7.5,align='center',color='muted')
    p.text(302,176,'keep only a quarter',size=7.5,align='center',color='muted')
    p.line(8,117,462,117,color='grid',width=.7)
    signs=[]
    for j,(heading,control,contrast) in enumerate((
        ('Train for the budget','Change training; fix random25','Random rollout: mixed − base'),
        ('Spend it well','Same observed test histories','Mixed model: risk − random'),
        ('Survive feedback','Predict → rebuild → predict','Rollout risk gap: mixed − base'))):
        left=8+j*156;p.rect(left,18,142,89,color='wash')
        p.text(left+8,97,heading,size=10.5,bold=True)
        p.text(left+8,83,control,control.replace('→',r'$\to$'),size=7.6,color='muted')
        tex=contrast.replace('−',r'$-$')
        p.text(left+8,71,contrast,tex,size=7.4,color='muted')
        for i,m in enumerate(MATERIALS):
            y=56-i*14;p.text(left+8,y,m,size=8)
            record=(data[m]['full']['random25']['training_effect'] if j==0 else
                    data[m]['observed']['risk_minus_random_mixed_training'] if j==1 else feedback[m]['record'])
            for seed,v in enumerate(values(record)):
                x=left+98+seed*12
                if v is None:p.cross(x,y,r=2.5)
                else:p.circle(x,y,r=2.8,color='teal' if v<0 else 'rust',filled=True)
                signs.append({'test':j,'material':m,'seed':seed,'value':v,'sign':None if v is None else (-1 if v<0 else 1)})
    p.circle(70,7,r=2.5,color='teal');p.text(78,7,'negative',size=7.5,color='muted')
    p.circle(145,7,r=2.5,color='rust');p.text(153,7,'positive',size=7.5,color='muted')
    p.cross(216,7,r=2.3);p.text(224,7,'graph limit',size=7.5,color='muted')
    p.text(320,7,'Each triplet: seeds 0, 1, 2',size=7.5,color='muted')
    caption=(r'\textbf{The goal is accurate forecasts with fewer additional interactions.} '
             r'The quarter-budget graph retains every native edge and only a quarter of the optional additions shown in the dense graph '
             r'(schematic; self-messages omitted). The cards separate training, placement on matched '
             r'observed test histories, and autonomous feedback. Triplets show seeds 0, 1, and 2 for '
             r'each named contrast; negative means lower error or a smaller risk-minus-random gap. '
             r"The Goop cross marks a missing complete-seed comparison: a seed-2 cached-risk rollout hit the candidate-pair limit.")
    product=p.save(HERE,'constructive_overview',caption,'fig:constructive-overview')
    return {'product':product,'seed_signs':signs,'concept_graph':{'native_pairs':native,'optional_pairs':optional,'selected_pairs':[(0,2),(4,7)],'nodes':coords}}


def atlas(data,feedback):
    p=Picture(470,330);records=[]
    starts=(120,237,354);width=102
    configs={
        'Goop':{'obs':(-11,11,1e10,(-10,0,10)),'full':(-4.6,1,10.,(-4,-2,0))},
        'WaterDrop':{'obs':(-4,4,1e10,(-4,0,4)),'full':(-2,1,100.,(-2,-1,0,1))},
        'Sand':{'obs':(-10,10,1e9,(-10,0,10)),'full':(-4,6,100.,(-4,0,6))}}
    for m,left in zip(MATERIALS,starts):
        p.text(left+width/2,315,m,size=11.5,bold=True,align='center')
        p.text(left+width/2,301,'100k from scratch' if m!='WaterDrop' else '100k → 110k paired',
               '100k from scratch' if m!='WaterDrop' else r'100k $\to$ 110k paired',size=7.7,align='center',color='muted')
    p.text(5,281,'Observed test',size=9,bold=True)
    p.text(5,157,'Full rollouts',size=9,bold=True)
    for y in (199,41):p.rect(2,y-11,461,19,color='wash')
    for row,(_,label) in enumerate(OBS):p.text(5,263-row*17,label,label.replace('Train:','Train:').replace('−',r'$-$'),size=7.9)
    for row,(_,label) in enumerate(POLICIES):p.text(5,139-row*17,label,size=8)
    p.text(5,37,'Change in risk gap',size=7.9,bold=True)

    def plot_record(material,section,key,record,row_y,left,config,state='predeclared'):
        low,high,scale,ticks=config
        def x(value):
            z=value*scale
            assert low<=z<=high,(material,section,key,z,low,high)
            return left+(z-low)/(high-low)*width
        if state=='not_predeclared':
            p.text(left+width/2,row_y,'Not evaluated',size=6.8,align='center',color='muted')
            records.append({'material':material,'section':section,'key':key,'state':state,'record':None});return
        vals=values(record);assert len(vals)==3
        item={'material':material,'section':section,'key':key,'state':state,'record':record,'scale':scale,'points':[]}
        mean,sd=record['mean'],record['sample_sd']
        if mean is not None:
            assert sd is not None and all(v is not None for v in vals)
            p.line(x(mean-sd),row_y+3,x(mean+sd),row_y+3,color='muted',width=.75)
            p.line(x(mean-sd),row_y+1.2,x(mean-sd),row_y+4.8,color='muted',width=.75)
            p.line(x(mean+sd),row_y+1.2,x(mean+sd),row_y+4.8,color='muted',width=.75)
            p.diamond(x(mean),row_y+3,r=2.25)
            item['mean_geometry']={'x':x(mean),'y':row_y+3,'sd_left':x(mean-sd),'sd_right':x(mean+sd)}
        else:
            assert sd is None and any(v is None for v in vals)
            p.text(left+width/2,row_y+3,'seed 2 incomplete',size=6.6,align='center',color='muted')
        for seed,v in enumerate(vals):
            if v is None:continue
            yy=row_y-3+(seed-1)*1.9
            p.circle(x(v),yy,r=1.7,color='teal' if v<0 else 'rust')
            item['points'].append({'seed':seed,'raw_value':v,'display_value':v*scale,'x':x(v),'y':yy})
        records.append(item)

    for m,left in zip(MATERIALS,starts):
        obsconf=configs[m]['obs'];fullconf=configs[m]['full']
        obs_exponent=data[m]['observed_display_exponent'];full_exponent=-1 if m=='Goop' else -2
        superscript=str.maketrans('0123456789-','⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
        p.text(left+width/2,282,'MSE ×10'+str(obs_exponent).translate(superscript),rf'MSE $\times10^{{{obs_exponent}}}$',size=7.8,align='center',color='muted')
        p.text(left+width/2,158,f'H{data[m]["horizon"]} · MSE ×10'+str(full_exponent).translate(superscript),rf'H{data[m]["horizon"]} $\cdot$ MSE $\times10^{{{full_exponent}}}$',size=7.8,align='center',color='muted')
        for conf,bottom,top,label_y in ((obsconf,186,273,179),(fullconf,28,148,21)):
            low,high,_,ticks=conf
            for tick in ticks:
                xx=left+(tick-low)/(high-low)*width
                p.line(xx,bottom,xx,top,color='zero' if tick==0 else 'grid',width=.8 if tick==0 else .35)
                p.text(xx,label_y,str(tick),size=7.4,align='center',color='muted')
        for row,(key,_) in enumerate(OBS):plot_record(m,'observed',key,data[m]['observed'][key],263-row*17,left,obsconf)
        for row,(key,_) in enumerate(POLICIES):
            item=data[m]['full'][key]
            plot_record(m,'full',key,item.get('training_effect'),139-row*17,left,fullconf,item['state'])
        plot_record(m,'full','risk_gap_interaction',feedback[m]['record'],37,left,fullconf)
    p.circle(12,5,r=1.7,color='teal');p.text(18,5,'seed effect < 0',r'seed effect $<0$',size=7.4,color='muted')
    p.circle(110,5,r=1.7,color='rust');p.text(116,5,'> 0',r'$>0$',size=7.4,color='muted')
    p.line(156,5,175,5,color='muted');p.diamond(165.5,5,r=2.2);p.text(182,5,'mean ± seed SD',r'mean $\pm$ seed SD',size=7.4,color='muted')
    p.text(308,5,'Goop: seed 2 reached the graph limit',size=6.7,color='muted')
    caption=(r'\textbf{Observed-test placement gains need not survive autonomous feedback.} '
             r'Training effects subtract base-only error from mixed-training error under the same policy; all lower policy rows show this effect. '
             r'Risk gaps subtract random-placement error from risk-placement error; their change compares the training arms. '
             r'Dots show paired seeds; diamonds and bars show mean $\pm$ sample seed SD, '
             r'not confidence intervals. Each panel has its own scale. In Goop, one mixed-training seed-2 trajectory '
             r'hit the candidate-pair limit under each of native, dense and cached-risk evaluation. These training effects and the change in risk gap '
             r'have no three-seed mean; the completed seed-0 and seed-1 points remain visible. '
             r'WaterDrop RMS was not evaluated because it was not included in that study. Observed risk scores the preceding observed '
             r'base graph; autonomous risk caches its own preceding graph.')
    product=p.save(HERE,'cross_material_evidence',caption,'fig:cross-material-evidence')
    # Appendix uses one column. Isolate the fixed-width native picture in an
    # explicit centered box; this hardens layout without changing coordinates.
    native=(HERE/'cross_material_evidence_picture.tex').read_text()
    wrapped='\n'.join([r'\begin{figure}[!htbp]', r'\centering',
        r'\noindent\makebox[\linewidth][c]{%',
        r'\begin{minipage}{470pt}', r'\noindent', native,
        r'\end{minipage}%', r'}', r'\caption{'+caption+'}',
        r'\label{fig:cross-material-evidence}', r'\end{figure}', ''])
    (HERE/'cross_material_evidence.tex').write_text(wrapped)
    product['sha256']['cross_material_evidence.tex']=hashlib.sha256(wrapped.encode()).hexdigest()
    (HERE/'cross_material_evidence_primitives.json').write_text(json.dumps(product,sort_keys=True,indent=2)+'\n')
    return {'product':product,'records':records,'axis_configs':configs}


def main():
    data,feedback=load()
    first=overview(data,feedback);second=atlas(data,feedback)
    assert len(first['seed_signs'])==27 and len(second['records'])==36
    assert sum(r['state']=='not_predeclared' for r in second['records'])==1
    undefined=[r for r in second['records'] if r['record'] is not None and r['record']['mean'] is None]
    assert len(undefined)==4 and all(r['material']=='Goop' for r in undefined)
    plotted=sum(len(r.get('points',[])) for r in second['records'])
    assert plotted==101
    # Presentation fidelity only: each displayed record is copied, never refitted or pooled.
    for r in second['records']:
        if r['record'] is None:continue
        src=(data[r['material']]['observed'][r['key']] if r['section']=='observed' else
             feedback[r['material']]['record'] if r['key']=='risk_gap_interaction' else data[r['material']]['full'][r['key']]['training_effect'])
        assert r['record']==src
        for point in r['points']:assert point['raw_value']==values(src)[point['seed']]
    for name,pin in PINS.items():assert hashlib.sha256((OLD/name).read_bytes()).hexdigest()==pin
    record={'overview_seed_outcomes':27,'atlas_statistic_slots':36,'plotted_seed_effects':plotted,'undefined_three_seed_means':4,'not_predeclared_slots':1}
    (HERE/'figure_counts.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    print(json.dumps(record,sort_keys=True))


if __name__=='__main__':main()
