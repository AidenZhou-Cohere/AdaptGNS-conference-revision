#!/usr/bin/env python3
"""Package-free inline picture export of the sealed qualitative renderer panels."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import textwrap
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RENDERER_SHA = "38937abd2474a212d0fecf515e00a00d778d56c4f477eb5e3d0a5a4f11d103eb"
RENDERER = HERE / "render_qualitative_v1.py"
if hashlib.sha256(RENDERER.read_bytes()).hexdigest() != RENDERER_SHA:
    raise RuntimeError("Sealed renderer bytes differ; review before export")
spec = importlib.util.spec_from_file_location("sealed_qualitative_renderer", RENDERER)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def escape(text):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(c, c) for c in str(text))


def basis(dims):
    if dims == 2:
        return np.eye(2)
    azimuth, elevation = np.deg2rad([-55., 25.])
    return np.array([[-np.sin(azimuth), np.cos(azimuth), 0.],
                     [-np.sin(elevation)*np.cos(azimuth), -np.sin(elevation)*np.sin(azimuth), np.cos(elevation)]])


def projection(bounds):
    dims = len(bounds)
    corners = np.asarray(list(itertools.product(*bounds)), dtype=float)
    center = bounds.mean(axis=1)
    matrix = basis(dims)
    projected = (corners-center) @ matrix.T
    low, high = projected.min(axis=0), projected.max(axis=0)
    scale = min(83./(high-low))
    def project(points):
        return (((points-center) @ matrix.T) - (high+low)/2) * scale + np.array([47., 43.])
    return project, corners, matrix


def number(value):
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def fragment(data):
    bounds, dims = data["bounds"], data["dims"]
    project, corners, matrix = projection(bounds)
    lines = ["% Inline generated figure; paste these bytes into the existing manuscript.",
             "% No input/includegraphics, packages, model calls, or particle subsampling.",
             r"\begin{figure*}[t]", r"\centering", r"\begingroup\setlength{\unitlength}{1pt}",
             r"\begin{picture}(480,427)"]
    def text(x, y, value, font="scriptsize", anchor=""):
        lines.append(r"\put("+number(x)+","+number(y)+r"){\makebox(0,0)"+("["+anchor+"]" if anchor else "")+
                     "{\\"+font+" "+escape(value)+"}}")
    def edge(a, b, thickness="0.2"):
        mid=(a+b)/2
        lines.append(r"\linethickness{"+thickness+r"pt}\qbezier("+
                     ",".join(map(number,a))+")("+",".join(map(number,mid))+")("+",".join(map(number,b))+")")
    def cross(x, y):
        lines.append(r"\put("+number(x)+","+number(y)+r"){\makebox(0,0){\tiny $\times$}}")
    title=("SYNTHETIC CHECK - " if data["synthetic"] else "")+f"{data['study']} | {data['updates']:,} training updates"
    text(240, 418, title, "small")
    text(240, 403, f"Fixed metadata median: source {data['source']['source_index']} | seed 0 | N={data['count']:,} | H={data['horizon']}")
    headings=(("Ground truth", "Saved target"), ("Ordinary-graph trained", "Base graph"),
              ("Mixture trained", "Random 25%"), ("Mixture trained", "Cached risk 25%"))
    for column, heading in enumerate(headings):
        text(66+116*column,383,heading[0]); text(66+116*column,373,heading[1])
    receipt={"schema":"qualitative_inline_picture_receipt_v1", "study":data["study"],
             "synthetic":data["synthetic"], "source_index":data["source"]["source_index"], "seed":0,
             "horizon":data["horizon"], "training_updates":data["updates"], "particles":data["count"],
             "renderer_sha256":RENDERER_SHA, "input_files_sha256":data["inputs_sha256"],
             "projection_matrix":matrix.tolist(), "bounds":bounds.tolist(), "panels":[],
             "figure_size_points":[480,427], "particle_subsampling":False, "external_image_dependencies":False,
             "compilation_verified":False,
             "projection_description":"Physical coordinates centered on metadata box, orthographically projected at elevation25/azimuth-55 for3D; equal physical scale. Same projection for every panel; no perspective or occlusion culling."}
    for row, step in enumerate((1,200,data["horizon"])):
        bottom=259-116*row
        text(240,bottom+97,f"Forecast {step} (source frame {step+5})", "tiny")
        for column in range(4):
            offset=np.array([19.+116*column, float(bottom)])
            if column==0:
                points=data["truth"].get(step)
                state=("SAVED TRUTH", "", "", "") if points is not None else ("MISSING SAVED TRUTH", "No saved accepted target", "", "")
                arm, policy, prefix=None,"truth",None
            else:
                panel=data["panels"][column-1]
                points=panel["frames"].get(step); state=renderer.panel_state(panel,step)
                arm,policy,prefix=panel["arm"],panel["policy"],panel["prefix"]
            if points is None:
                # Omit the empty 3D box so the failure label is unobstructed.
                lines.append(r"\put("+number(offset[0]+3)+","+number(offset[1]+11)+r"){\framebox(88,66){}}")
                words=[part for item in state[:3] if item for part in textwrap.wrap(item.replace("_"," "),25)]
                for index, word in enumerate(words):
                    text(offset[0]+47, offset[1]+44+(len(words)-1)*4-index*8, word, "tiny")
                outside=None; count=0
            else:
                for a,b in itertools.combinations(corners,2):
                    if np.count_nonzero(a!=b)==1:
                        pa,pb=project(np.stack([a,b]))+offset
                        edge(pa,pb)
                physical_origin=bounds[:,0]
                for axis in range(dims):
                    endpoint=physical_origin.copy(); endpoint[axis]=bounds[axis,1]
                    position=project(endpoint[None,:])[0]+offset
                    shift=np.array([0.,-7.]) if dims==2 and axis==0 else np.array([5.,1.])
                    text(*(position+shift), "xyz"[axis], "tiny")
                out=np.any((points<bounds[:,0]) | (points>bounds[:,1]),axis=1)
                coordinates=project(np.clip(points,bounds[:,0],bounds[:,1]))+offset
                # Small solid circle glyphs are legible at the native 480pt figure width.
                diameter="1.4" if data["count"]<200 else "0.8"
                for position,is_outside in zip(coordinates,out):
                    if is_outside:
                        cross(*position)
                    else:
                        lines.append(r"\put("+number(position[0])+","+number(position[1])+r"){\circle*{"+diameter+"}}")
                outside=int(out.sum());count=len(points)
                text(offset[0]+47,offset[1]+89,f"Outside box: {outside}/{data['count']}","tiny")
                if column and data["panels"][column-1]["status"]=="failed":
                    text(offset[0]+47,offset[1]-4,f"Run failed; accepted prefix {prefix}","tiny")
            receipt["panels"].append({"arm":arm,"policy":policy,"forecast_step":step,"source_frame":step+5,
                "state":state[0],"details":list(state[1:]),"accepted_prefix":prefix,"outside_box":outside,
                "particle_glyphs":count})
    text(240,10,"Filled circles: inside metadata box. Crosses: outside box, marker clamped to boundary.","tiny")
    lines.extend([r"\end{picture}",r"\endgroup"])
    caption=("Synthetic fixture only. " if data["synthetic"] else "") + (
        f"{data['study']} qualitative rollout at {data['updates']:,} training updates, training seed 0. "
        "The fixed source is the lower median of all test sources sorted by initial particle count and original source index, "
        "selected without policy errors. All particles are displayed at forecasts 1, 200 and H, on common metadata bounds "
        + "("+", ".join(f"{'xyz'[d]}=[{number(lo)}, {number(hi)}]" for d,(lo,hi) in enumerate(bounds))+"). "
        "Out-of-box particles are counted and marked by crosses clamped to the box. "
        + ("The orthographic view uses elevation 25 and azimuth -55 degrees with equal physical scaling. " if dims==3 else "")
        + "Missing and failed frames retain their recorded state and accepted prefix; unreturned prefixes remain unknown. "
        "Ground truth uses agreeing saved targets. Cached risk is the mix/laggedrisk25 rollout policy, distinct from the "
        "previous-observed same-state diagnostic. This single example does not estimate population performance or isolate dimensionality as a cause.")
    lines.extend([r"\caption{"+escape(caption)+"}",r"\label{fig:qualitative-"+data["study"].lower()+"}",r"\end{figure*}"])
    receipt["caption"]=caption
    receipt["total_particle_glyphs"]=sum(panel["particle_glyphs"] for panel in receipt["panels"])
    return "\n".join(lines)+"\n",receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--output-prefix",type=Path,required=True)
    args=parser.parse_args()
    prefix=args.output_prefix.resolve()
    renderer.require(HERE in prefix.parents,"Output must remain under presentation work directory")
    tex=Path(str(prefix)+".tex");receipt_path=Path(str(prefix)+".json")
    renderer.require(not tex.exists() and not receipt_path.exists(),"Fresh output prefix required")
    started=time.perf_counter()
    data=renderer.load_request(args.request)
    content,receipt=fragment(data)
    receipt.update(exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   fragment_sha256=hashlib.sha256(content.encode()).hexdigest(),fragment_bytes=len(content.encode()),
                   parse_and_export_seconds=time.perf_counter()-started)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    tex.write_text(content);receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:receipt[k] for k in ("synthetic","fragment_bytes","total_particle_glyphs","parse_and_export_seconds")},sort_keys=True))


if __name__=="__main__":
    main()
