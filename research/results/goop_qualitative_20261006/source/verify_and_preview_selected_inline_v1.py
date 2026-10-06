"""Check the selected source's inline glyphs independently and save PNG only.

Reads the exact admitted selected NPZs; no model/evaluator/renderer is imported.
The PNG is a direct drawing of the emitted picture commands, not a TeX proof.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "adaptgns-goop-qualitative-mpl-v1"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PREFIX = HERE / "goop_fixed_source12_inline_v1"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    request_path = HERE / "render_request_v1.json"
    request = json.loads(request_path.read_text())
    fetch = json.loads((HERE / "fetch_manifest_v1.json").read_text())
    for item in fetch["files"]:
        assert sha(Path(item["local_path"])) == item["sha256"]
    receipt = json.loads(PREFIX.with_suffix(".json").read_text())
    tex = PREFIX.with_suffix(".tex").read_text()
    assert sha(PREFIX.with_suffix(".tex")) == receipt["fragment_sha256"]
    assert not re.search(r"\\(?:input|include|includegraphics|documentclass|usepackage|RequirePackage|special|directlua|write)\b", tex)
    depth = 0
    for token in re.findall(r"\\.|[{}]", tex):
        depth += int(token == "{") - int(token == "}")
        assert depth >= 0
    assert depth == 0
    manifest = json.loads((HERE / request["manifest"]["path"]).read_text())
    ordered = sorted(manifest["records"], key=lambda row: (row["positions"]["shape"][1], row["source_index"]))
    selected = ordered[(len(ordered)-1)//2]
    assert selected["source_index"] == receipt["source_index"] == 12
    assert selected["positions"]["shape"] == [401, 1083, 2]
    assert receipt["seed"] == 0 and receipt["horizon"] == 395
    bounds = np.asarray(manifest["metadata"]["bounds"], dtype=np.float64)
    arrays = []
    initial = types = None
    for cell in request["cells"]:
        row_path = HERE / cell["row"]["path"]
        row = json.loads(row_path.read_text())
        protocol_path = HERE / cell["protocol"]["path"]
        assert sha(row_path) == cell["row"]["sha256"]
        assert sha(protocol_path) == cell["protocol"]["sha256"] == row["protocol_sha256"]
        assert row["source_index"] == 12 and row["training_seed"] == 0 and row["horizon"] == 395
        assert row["status"] == "complete" and row["completed_steps"] == 395
        trace = row_path.parent / row["trace_file"]
        assert sha(trace) == row["trace_sha256"]
        with np.load(trace, allow_pickle=False) as saved:
            values = {key: saved[key] for key in ("forecast_steps", "predicted_positions", "ground_truth_positions", "initial_observed_positions", "particle_types", "bounds")}
        assert np.array_equal(values["bounds"], bounds)
        if initial is None:
            initial, types = values["initial_observed_positions"], values["particle_types"]
        assert np.array_equal(initial, values["initial_observed_positions"])
        assert np.array_equal(types, values["particle_types"])
        arrays.append(values)

    point_re = re.compile(r"\\put\(([-\d.]+),([-\d.]+)\)\{(?:\\circle\*\{(0\.8)\}|\\makebox\(0,0\)\{\\tiny \$\\times\$\})\}")
    emitted = [(float(x), float(y), bool(diameter)) for x, y, diameter in point_re.findall(tex)]
    assert len(emitted) == receipt["total_particle_glyphs"] == 12996
    cursor = 0
    panels = []
    for row_index, step in enumerate((1, 200, 395)):
        ixs = [int(np.flatnonzero(value["forecast_steps"] == step)[0]) for value in arrays]
        truth = arrays[0]["ground_truth_positions"][ixs[0]]
        assert all(np.array_equal(truth, value["ground_truth_positions"][ix]) for value, ix in zip(arrays, ixs))
        points_list = [truth] + [value["predicted_positions"][ix] for value, ix in zip(arrays, ixs)]
        for column, points in enumerate(points_list):
            assert points.shape == (1083, 2) and np.isfinite(points).all()
            outside = np.any((points < bounds[:, 0]) | (points > bounds[:, 1]), axis=1)
            # Independent closed-form specialization of this fixed 2D box.
            expected = (np.clip(points, bounds[:, 0], bounds[:, 1]) - .5) * (83/.8)
            expected += np.asarray([47 + 19 + 116*column, 43 + 259 - 116*row_index])
            actual = np.asarray(emitted[cursor:cursor+1083])
            assert np.allclose(actual[:, :2], expected, rtol=0, atol=.000500001)
            assert np.array_equal(actual[:, 2].astype(bool), ~outside)
            recorded = receipt["panels"][4*row_index+column]
            assert recorded["forecast_step"] == step and recorded["source_frame"] == step+5
            assert recorded["particle_glyphs"] == 1083 and recorded["outside_box"] == int(outside.sum())
            panels.append({"forecast_step": step, "policy": recorded["policy"], "outside_box": int(outside.sum()),
                           "all_1083_particle_glyphs_checked": True, "maximum_coordinate_rounding_error_pt": float(np.max(np.abs(actual[:, :2]-expected)))})
            cursor += 1083

    # Render emitted commands directly. This creates only PNG, without TeX or PDF.
    fig = plt.figure(figsize=(480/72,427/72), dpi=200)
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,480); ax.set_ylim(0,427); ax.axis("off")
    plt.rcParams.update({"font.family": "DejaVu Serif"})
    for line in tex.splitlines():
        q = re.fullmatch(r"\\linethickness\{([\d.]+)pt\}\\qbezier\(([-\d.]+),([-\d.]+)\)\(([-\d.]+),([-\d.]+)\)\(([-\d.]+),([-\d.]+)\)", line)
        if q:
            thickness,x1,y1,xm,ym,x2,y2 = map(float,q.groups())
            assert np.allclose([(x1+x2)/2,(y1+y2)/2],[xm,ym],atol=.00051)
            ax.plot([x1,x2],[y1,y2],color="black",linewidth=thickness,zorder=1)
        t = re.fullmatch(r"\\put\(([-\d.]+),([-\d.]+)\)\{\\makebox\(0,0\)\{\\(small|scriptsize|tiny) (.*)\}\}", line)
        if t and t.group(4) != r"$\times$":
            x,y,size,label=t.groups();label=label.replace(r"\%","%").replace(r"\_","_").replace(r"\&","&")
            ax.text(float(x),float(y),label,ha="center",va="center",fontsize={"small":9,"scriptsize":7,"tiny":5}[size],zorder=4)
    points=np.asarray(emitted);inside=points[:,2].astype(bool)
    ax.scatter(points[inside,0],points[inside,1],s=.8**2,c="black",linewidths=0,zorder=2)
    ax.scatter(points[~inside,0],points[~inside,1],s=2.6**2,c="black",marker="x",linewidths=.35,zorder=3)
    png=HERE/'goop_fixed_source12_inline_direct_preview_v1.png'
    assert not png.exists(), "Preserve existing preview"
    fig.savefig(png,dpi=200);plt.close(fig)
    for item in fetch["files"]:assert sha(Path(item["local_path"]))==item["sha256"]
    result={"schema":"selected_goop_inline_independent_check_v1","source_index":12,"seed":0,"forecasts":[1,200,395],
            "all_input_hashes_rechecked":True,"glyphs_checked":cursor,"panels":panels,"truth_and_initial_identity_checked":True,
            "fragment_sha256":sha(PREFIX.with_suffix('.tex')),"request_sha256":sha(request_path),"preview_png_sha256":sha(png),
            "raw_array_reads":"Only the three fetched admitted selected trace NPZs; six named arrays per trace.",
            "no_model_or_evaluator_import":True,"no_remote_action":True,"no_tex_compiler_or_pdf_or_tab":True,
            "preview_scope":"Direct drawing of emitted picture coordinates and labels; fonts approximate native TeX. Root must inspect actual integrated output.",
            "outside_scope":"Any strict metadata-box crossing, not the quantitative >1e-6 threshold; crosses clamp positions and do not encode excursion magnitude."}
    (HERE/'selected_inline_verification_v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({"glyphs_checked":cursor,"panels":panels,"png":str(png)},indent=2))


if __name__=="__main__":main()
