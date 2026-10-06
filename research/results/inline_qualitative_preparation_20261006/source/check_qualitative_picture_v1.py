#!/usr/bin/env python3
"""Synthetic-only invariant and source-volume checks; deliberately no TeX compile."""
import copy
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import export_qualitative_picture_v1 as exporter


def check_braces(content):
    depth=0
    for token in re.findall(r"\\.|[{}]",content):
        if token=="{":depth+=1
        elif token=="}":depth-=1
        assert depth>=0,"Closing brace without opener"
    assert depth==0,"Unclosed brace"


def main():
    checks=[]
    for study,horizon,glyphs in (("Goop2D",395,231),("Goop3D",295,297)):
        request=exporter.HERE/'qualitative_synthetic_v3'/study/'request.json'
        data=exporter.renderer.load_request(request)
        content,receipt=exporter.fragment(data)
        check_braces(content)
        assert not re.search(r"\\(?:input|include|includegraphics|documentclass|usepackage|RequirePackage|special|directlua|write)\b",content)
        assert content.count(r"\begin{figure*}")==content.count(r"\end{figure*}")==1
        assert content.count(r"\begin{picture}")==content.count(r"\end{picture}")==1
        assert receipt['source_index']==1 and receipt['seed']==0
        assert len(receipt['panels'])==12 and receipt['total_particle_glyphs']==glyphs
        assert content.count(r"\circle*{")+content.count(r"\tiny $\times$")==glyphs
        assert {p['forecast_step'] for p in receipt['panels']}=={1,200,horizon}
        assert all(p['source_frame']==p['forecast_step']+5 for p in receipt['panels'])
        assert {p['policy'] for p in receipt['panels']}=={'truth','base','random25','laggedrisk25'}
        assert next(p for p in receipt['panels'] if p['policy']=='base' and p['forecast_step']==horizon)['outside_box']==2
        cached=[p for p in receipt['panels'] if p['policy']=='laggedrisk25']
        assert all(p['accepted_prefix'] is None for p in cached) if study=='Goop2D' else all(p['accepted_prefix']==0 for p in cached)
        matrix=exporter.basis(data['dims'])
        assert np.allclose(matrix@matrix.T,np.eye(2),atol=1e-14)
        project,corners,_=exporter.projection(data['bounds'])
        projected=project(corners)
        assert np.all(projected>=[0,0]) and np.all(projected<=[94,86])
        checks.append({'study':study,'status':'passed','glyphs':glyphs,'fragment_bytes':len(content.encode())})
    assert exporter.escape(r'100% _ {x} \input{bad}')==r'100\% \_ \{x\} \textbackslash{}input\{bad\}'
    data=copy.deepcopy(data)
    n=10000
    for step,points in data['truth'].items():data['truth'][step]=np.tile(points,(304,1))[:n]
    for panel in data['panels']:
        for step,points in panel['frames'].items():panel['frames'][step]=np.tile(points,(304,1))[:n]
    data['count']=n
    data['source']['positions']['shape'][1]=n
    started=time.perf_counter();content,receipt=exporter.fragment(data);elapsed=time.perf_counter()-started
    assert receipt['total_particle_glyphs']==90000
    assert content.count(r"\circle*{")+content.count(r"\tiny $\times$")==90000
    check_braces(content)
    record={'schema':'qualitative_picture_synthetic_checks_v1','checks':checks,'escaped_tex_text':'passed',
            'projection_basis_orthonormal':'passed','package_and_image_dependencies':'none',
            'latex_compiler_called':False,'visual_compilation_verified':False,
            'synthetic_stress':{'particles_per_panel':n,'populated_panels':9,'total_particle_glyphs':90000,
                                'export_seconds':elapsed,'fragment_bytes':len(content.encode()),
                                'method':'Repeat inert synthetic particle coordinates, same IDs/order across every panel; stress only, not research evidence.'},
            'exporter_sha256':hashlib.sha256(Path(exporter.__file__).read_bytes()).hexdigest(),
            'renderer_sha256':hashlib.sha256(exporter.RENDERER.read_bytes()).hexdigest()}
    assert record['renderer_sha256']==exporter.RENDERER_SHA
    destination=exporter.HERE/'qualitative_picture_checks_v1.json'
    assert not destination.exists(),'Preserve prior check record'
    destination.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
