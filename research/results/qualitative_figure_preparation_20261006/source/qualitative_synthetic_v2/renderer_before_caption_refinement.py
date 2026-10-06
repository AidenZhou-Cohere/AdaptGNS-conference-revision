#!/usr/bin/env python3
"""Render the fixed v2 qualitative example from hashed, saved rollout artifacts.

No model, raw position file, evaluator, or inference code is imported. Root must
separately admit the study. See qualitative_renderer_v1.md for the request schema.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
PLAN = HERE / "qualitative_selection_plan_v2.md"
STUDIES = {"Goop2D": ("Goop", 2, 395, 100000),
           "Goop3D": ("Goop-3D", 3, 295, 25000),
           "Sand": ("Sand", 2, 314, 100000)}
PROTOCOL_SCHEMAS = {"Goop2D": "adaptgns_goop_graph_support_final_evaluation_v1",
                    "Goop3D": "adaptgns_goop3d_graph_support_evaluation_v1",
                    "Sand": "adaptgns_sand_graph_support_final_evaluation_v1"}
METHODS = (("base", "base", "Ordinary-graph trained\nBase graph"),
           ("mix", "random25", "Mixture trained\nRandom 25%"),
           ("mix", "laggedrisk25", "Mixture trained\nCached risk 25%"))
MISSING_STATES = {"timed_out_current", "not_completed_before_invocation_end",
                  "never_started", "outside_evaluated_source_grid"}
BLUE, RED, GREY = "#17658a", "#b53636", "#656565"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_bound(ref, base, inputs):
    """Hash exactly the bytes that will be parsed, avoiding a second file read."""
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"},
            "Each file reference requires path and sha256 only")
    path = (base / ref["path"]).resolve()
    data = path.read_bytes()
    require(sha(data) == ref["sha256"], f"SHA256 mismatch: {path}")
    inputs[str(path)] = ref["sha256"]
    return path, data


def select_source(manifest, study):
    dataset, dims, horizon, _ = STUDIES[study]
    require(manifest.get("dataset") == dataset and manifest.get("split") == "test",
            "Study requires its exact test manifest")
    records = manifest.get("records", [])
    require(len(records) > 0 and manifest.get("record_count") == len(records),
            "Full source metadata census required")
    for i, record in enumerate(records):
        shape = record.get("positions", {}).get("shape", [])
        require(record.get("source_index") == i and len(shape) == 3
                and shape[0] == horizon + 6 and shape[2] == dims
                and type(shape[1]) is int and shape[1] > 0,
                "Ordered source indices and correct position shape required")
    ordered = sorted(records, key=lambda r: (r["positions"]["shape"][1], r["source_index"]))
    return ordered[(len(ordered) - 1) // 2]


def load_request(request_path):
    request_path = Path(request_path).resolve()
    request_bytes = request_path.read_bytes()
    request = json.loads(request_bytes)
    require(request.get("schema") == "qualitative-render-request-v1", "Unknown request schema")
    require(request.get("study") in STUDIES, "Unknown study")
    require(type(request.get("synthetic")) is bool, "Explicit synthetic flag required")
    if not request["synthetic"]:
        require(request.get("root_admission_note", "").strip(),
                "A root admission note is required; renderer cannot admit experiments")
    study = request["study"]
    _, dims, horizon, updates = STUDIES[study]
    inputs = {str(request_path): sha(request_bytes)}
    _, plan_bytes = read_bound(request["selection_plan"], request_path.parent, inputs)
    require(plan_bytes == PLAN.read_bytes(), "Only the fixed v2 selection plan is supported")
    manifest_path, data = read_bound(request["manifest"], request_path.parent, inputs)
    manifest = json.loads(data)
    source = select_source(manifest, study)
    count = source["positions"]["shape"][1]
    bounds = np.asarray(manifest["metadata"]["bounds"], dtype=np.float64)
    require(bounds.shape == (dims, 2) and np.isfinite(bounds).all()
            and np.all(bounds[:, 1] > bounds[:, 0]), "Finite physical metadata bounds required")
    cells = request.get("cells", [])
    require(len(cells) == len(METHODS) and
            [(c.get("arm"), c.get("policy")) for c in cells] == [m[:2] for m in METHODS],
            "Exactly base/base, mix/random25, mix/laggedrisk25 in fixed order required")
    panels, truth, shared_initial, shared_types = [], {}, None, None
    checkpoints = {}
    for cell, (arm, policy, title) in zip(cells, METHODS):
        panel = {"arm": arm, "policy": policy, "title": title, "frames": {}, "prefix": None}
        if "row" not in cell:
            require(cell.get("coverage_state") in MISSING_STATES
                    and isinstance(cell.get("reason"), str) and cell["reason"].strip()
                    and "protocol" not in cell, "Missing cell requires coverage state and reason, without row/protocol")
            if cell["coverage_state"] == "outside_evaluated_source_grid":
                population = len(manifest["records"])
                grid = set(range(population)) if population <= 30 else {
                    j * (population - 1) // 29 for j in range(30)}
                require(study == "Goop3D" and source["source_index"] not in grid,
                        "Outside-grid missing state requires a D3 source excluded by its declared grid")
            panel.update(status="missing", reason=cell["reason"], coverage_state=cell["coverage_state"])
            panels.append(panel)
            continue
        require("coverage_state" not in cell and "reason" not in cell,
                "Committed rows must not override their recorded outcome")
        row_path, raw = read_bound(cell["row"], request_path.parent, inputs)
        row = json.loads(raw)
        _, protocol_raw = read_bound(cell["protocol"], request_path.parent, inputs)
        protocol = json.loads(protocol_raw)
        require(row.get("protocol_sha256") == sha(protocol_raw), "Row/protocol binding differs")
        require(request["manifest"]["sha256"] in protocol.get("input_files_sha256", {}).values(),
                "Protocol must bind the selected manifest bytes")
        require(protocol.get("schema") == PROTOCOL_SCHEMAS[study]
                and protocol.get("mode") == "full-rollout" and protocol.get("split") == "test",
                "Only saved final test full-rollout traces are supported")
        require(row.get("objective") == "faithful" and
                (protocol.get("purpose") == "final_evaluation" if study == "Goop3D"
                 else protocol.get("model", {}).get("objective") == "faithful"),
                "Faithful final scientific study required")
        model = protocol if study == "Goop3D" else protocol.get("model", {})
        endpoint = model.get("checkpoint_updates") if study == "Goop3D" else model.get("completed_updates")
        require(endpoint == updates and model.get("arm") == arm and model.get("seed") == 0,
                "Wrong training budget, arm or seed; only fixed seed0 is supported")
        checkpoint = model.get("checkpoint_sha256", "")
        require(len(checkpoint) == 64 and all(c in "0123456789abcdef" for c in checkpoint),
                "Checkpoint identity is required, without opening its contents")
        require(arm not in checkpoints or checkpoints[arm] == checkpoint,
                "Policies within one training arm must use the same checkpoint")
        checkpoints[arm] = checkpoint
        require(row.get("arm") == arm and row.get("policy") == policy
                and type(row.get("training_seed")) is int and row["training_seed"] == 0
                and row.get("source_index") == source["source_index"]
                and row.get("trajectory_id") == source["id"] and row.get("particles") == count
                and row.get("horizon") == horizon, "Row differs from fixed example identity")
        status, prefix, failure = row.get("status"), row.get("completed_steps"), row.get("failure")
        require(type(prefix) is int and 0 <= prefix <= horizon
                and ((status == "complete" and prefix == horizon and failure is None)
                     or (status == "failed" and prefix < horizon and isinstance(failure, dict))),
                "Inconsistent committed status or accepted prefix")
        key = "artifact" if study == "Goop3D" else "trace"
        trace_name = row.get(key + "_file", "")
        require(trace_name and Path(trace_name).name == trace_name, "Trace must be a sibling filename")
        _, trace_raw = read_bound({"path": trace_name, "sha256": row[key + "_sha256"]}, row_path.parent, inputs)
        with np.load(io.BytesIO(trace_raw), allow_pickle=False) as saved:
            # Never load rejected_prediction, failed_input_history or rejected_history.
            steps = saved["forecast_steps"]
            predicted, actual = saved["predicted_positions"], saved["ground_truth_positions"]
            initial, types = saved["initial_observed_positions"], saved["particle_types"]
            require(steps.ndim == 1 and np.issubdtype(steps.dtype, np.integer)
                    and np.all(np.diff(steps) > 0) and np.all((steps >= 1) & (steps <= prefix))
                    and set(steps.tolist()) <= {1, 10, 50, 200, horizon}, "Invalid accepted forecast IDs")
            require(predicted.shape == actual.shape == (len(steps), count, dims)
                    and np.isfinite(predicted).all() and np.isfinite(actual).all(), "Invalid accepted trace arrays")
            require(initial.shape == (6, count, dims) and types.shape == (count,)
                    and np.array_equal(saved["bounds"], bounds), "Trace shape/bounds differ")
            if shared_initial is None:
                shared_initial, shared_types = initial.copy(), types.copy()
            else:
                require(np.array_equal(initial, shared_initial) and np.array_equal(types, shared_types),
                        "Policies must preserve identical particle IDs and initial observations")
            for index, step in enumerate(steps.tolist()):
                if step in truth:
                    require(np.array_equal(actual[index], truth[step]), "Saved ground truth differs between policies")
                truth[step] = actual[index].copy()
                panel["frames"][step] = predicted[index].copy()
        panel.update(status=status, prefix=prefix, reason=(failure or {}).get("category", ""),
                     checkpoint_sha256=checkpoint)
        panels.append(panel)
    # Truth is taken only from agreeing accepted saved frames, never from raw data.
    return {"study": study, "dims": dims, "horizon": horizon, "updates": updates,
            "synthetic": request["synthetic"], "source": source, "count": count, "bounds": bounds,
            "panels": panels, "truth": truth, "inputs_sha256": inputs,
            "root_admission_note": request.get("root_admission_note")}


def panel_state(panel, step):
    prefix = panel["prefix"]
    if panel["status"] == "missing":
        return "MISSING", "Accepted prefix: unknown", panel["coverage_state"], panel["reason"]
    if step in panel["frames"]:
        if panel["status"] == "failed":
            return "SAVED PREFIX", f"Run failed; accepted {prefix}", panel["reason"], ""
        return "SAVED", f"Complete: {prefix} forecasts", "", ""
    if panel["status"] == "failed" and step > prefix:
        return "FAILED BEFORE FRAME", f"Accepted prefix: {prefix}", panel["reason"], ""
    return "MISSING SAVED FRAME", f"Accepted prefix: {prefix}", "Artifact lacks requested forecast", ""


def plot_particles(ax, points, bounds, dims):
    outside = np.any((points < bounds[:, 0]) | (points > bounds[:, 1]), axis=1)
    inside_points = points[~outside]
    size = max(.65, min(9., 750. / len(points)))
    coordinates = [inside_points[:, d] for d in range(dims)]
    options = {"depthshade": False} if dims == 3 else {}
    ax.scatter(*coordinates, s=size, color=BLUE, linewidths=0, rasterized=True, **options)
    if outside.any():
        clipped = np.clip(points[outside], bounds[:, 0], bounds[:, 1])
        ax.scatter(*[clipped[:, d] for d in range(dims)], s=14, color=RED, marker="x",
                   linewidths=.7, rasterized=True, **options)
    return int(outside.sum())


def render(data, output_prefix):
    output_prefix = Path(output_prefix).resolve()
    require(HERE in output_prefix.parents, "Outputs must remain under the presentation work directory")
    pdf, receipt_path = Path(str(output_prefix) + ".pdf"), Path(str(output_prefix) + ".json")
    require(not pdf.exists() and not receipt_path.exists(), "Fresh output prefix required; preserve earlier artifacts")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    dims, bounds, horizon = data["dims"], data["bounds"], data["horizon"]
    fig = plt.figure(figsize=(10.8, 8.9))
    grid_top, grid_bottom = .81, .14
    grid = fig.add_gridspec(3, 4, left=.08, right=.935 if dims == 3 else .98,
                           bottom=grid_bottom, top=grid_top, wspace=.30, hspace=.40)
    label = "SYNTHETIC CHECK - " if data["synthetic"] else ""
    fig.suptitle(f"{label}{data['study']} | {data['updates']:,} training updates", y=.98, fontsize=13, weight="bold")
    fig.text(.5, .935, f"Fixed metadata median: source {data['source']['source_index']} | training seed 0 | "
             f"{data['count']:,} particles | H = {horizon}", ha="center", fontsize=9)
    fig.text(.5, .908, "All particles shown. Shared physical bounds; red crosses mark out-of-box positions at the box edge."
             + ("\n3D: fixed orthographic view (elevation 25 degrees, azimuth -55 degrees)." if dims == 3 else ""),
             ha="center", va="top", fontsize=8, color=GREY)
    receipt = {k: data[k] for k in ("study", "horizon", "updates", "synthetic", "count", "inputs_sha256", "root_admission_note")}
    receipt.update(schema="qualitative-render-receipt-v1", source_index=data["source"]["source_index"], seed=0,
                   selection="lower median of full test census sorted by (initial particle count, original source index)",
                   bounds=bounds.tolist(), display_steps=[1, 200, horizon], panels=[],
                   view={"kind": "orthographic", "elevation": 25, "azimuth": -55} if dims == 3 else {"kind": "physical xy"},
                   renderer_sha256=sha(Path(__file__).read_bytes()),
                   caveat="One trajectory illustration, not population evidence or a causal dimensionality comparison.")
    for r, step in enumerate((1, 200, horizon)):
        for c in range(4):
            ax = fig.add_subplot(grid[r, c], projection="3d" if dims == 3 else None)
            ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1])
            ax.set_xlabel("x", labelpad=1); ax.set_ylabel("y", labelpad=1)
            ax.tick_params(labelsize=6, pad=1)
            if dims == 3:
                ax.set_zlim(*bounds[2]); ax.set_zlabel("z", labelpad=1)
                ax.set_proj_type("ortho"); ax.view_init(elev=25, azim=-55)
                ax.set_box_aspect(bounds[:, 1] - bounds[:, 0])
            else:
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=.15, linewidth=.5)
            if r == 0:
                ax.set_title("Ground truth\nSaved target" if c == 0 else data["panels"][c-1]["title"], fontsize=9, pad=10)
            text_fn = ax.text2D if dims == 3 else ax.text
            if c == 0:
                points = data["truth"].get(step)
                state = ("SAVED TRUTH", "", "", "") if points is not None else ("MISSING SAVED TRUTH", "No accepted trace saves this target", "", "")
                policy, arm, prefix = "truth", None, None
            else:
                panel = data["panels"][c-1]
                points, state = panel["frames"].get(step), panel_state(panel, step)
                policy, arm, prefix = panel["policy"], panel["arm"], panel["prefix"]
            out_count = plot_particles(ax, points, bounds, dims) if points is not None else None
            if points is None:
                ax.set_facecolor("#f7f3f2")
                message = "\n".join(textwrap.fill(line.replace("_", " "), 25) for line in state if line)
                text_fn(.5, .5, message, transform=ax.transAxes, ha="center", va="center", fontsize=8, color=RED,
                        bbox={"facecolor": "#fffafa", "edgecolor": "none", "pad": 2})
            else:
                note = f"Outside box: {out_count}/{data['count']}"
                if c and data["panels"][c-1]["status"] == "failed":
                    note += f"\nRun failed; accepted prefix {prefix}"
                text_fn(.02, .98, note, transform=ax.transAxes, va="top", fontsize=6.5,
                        color=RED if out_count or (c and data["panels"][c-1]["status"] == "failed") else GREY,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1})
            receipt["panels"].append({"forecast_step": step, "original_source_frame": step + 5, "arm": arm,
                "policy": policy, "state": state[0], "accepted_prefix": prefix, "outside_box": out_count,
                "details": list(state[1:]), "particles_plotted": data["count"] if points is not None else 0})
        fig.text(.018, grid_top - (r+.5)*((grid_top-grid_bottom)/3), f"Forecast {step}\nSource frame {step+5}",
                 ha="left", va="center", rotation=90, fontsize=8, weight="bold")
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ls="", markersize=4, label="Particle inside metadata box"),
                        Line2D([], [], color=RED, marker="x", ls="", markersize=5, label="Particle outside box (marker clamped)")],
               loc="lower center", bbox_to_anchor=(.5, .081), ncol=2, frameon=False, fontsize=8)
    fig.text(.5, .045, "Fixed example selected without policy errors. Missing and failed frames remain visible.\n"
             "Cached risk 25% = mix / laggedrisk25; it is distinct from the previous-observed same-state diagnostic.\n"
             "Truth is recovered only from agreeing saved targets. These panels do not estimate population performance.",
             ha="center", va="center", fontsize=7.4, color=GREY)
    fig.savefig(pdf, format="pdf", dpi=200, metadata={"Title": f"{label}{data['study']} fixed qualitative trajectory", "CreationDate": None})
    plt.close(fig)
    receipt["pdf_sha256"] = sha(pdf.read_bytes())
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    receipt = render(load_request(args.request), args.output_prefix)
    print(json.dumps({"study": receipt["study"], "source_index": receipt["source_index"],
                      "synthetic": receipt["synthetic"], "pdf_sha256": receipt["pdf_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
