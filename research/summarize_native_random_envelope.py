"""Strict complete-cohort saved-array summary; no model inference or fitting."""
import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from research import native_random_envelope as evaluate

full, bridge, native = evaluate.full, evaluate.bridge, evaluate.native
IDENTITY = ("split", "source_index", "target_frame", "trajectory_id")
BOUNDARIES = ("fraction_particles_outside", "fraction_particles_outside_by_more_than_1e-6",
              "maximum_coordinate_excursion", "mean_particle_maximum_excursion")


class Audit:
    def __init__(self): self.checks = 0
    def check(self, condition, message):
        self.checks += 1
        evaluate.require(condition, message)
    def equal(self, a, b, message): self.check(exact_value(a, b), message)
    def array(self, a, b, message): self.check(np.array_equal(a, b, equal_nan=True), message)
    def scalar(self, a, b, message):
        self.check(type(a) in (float, int) and np.isfinite(a) and np.isclose(a, b, rtol=1e-12, atol=1e-16), message)


def exact_value(a, b):
    """JSON equality without Python's bool-equals-integer shortcut."""
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and set(a)==set(b) and all(exact_value(a[k],b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a)==len(b) and all(exact_value(x,y) for x,y in zip(a,b))
    return a == b


def complete_mean(values):
    evaluate.require(all(v is None or (type(v) in (float,int) and np.isfinite(v)) for v in values),
                     "Finite nonboolean scalar or explicit null required")
    return float(np.mean(values)) if values and all(v is not None and np.isfinite(v) for v in values) else None


def seed_statistics(values):
    evaluate.require(len(values) == 3, "Exactly three trained seeds required")
    evaluate.require(all(v is None or (type(v) in (float,int) and np.isfinite(v)) for v in values),
                     "Finite nonboolean seed scalar or explicit null required")
    good = all(v is not None and np.isfinite(v) for v in values)
    return {"seed_values": values, "required_seeds": 3, "defined_seeds": sum(v is not None for v in values),
        "mean": float(np.mean(values)) if good else None, "sample_sd": float(np.std(values, ddof=1)) if good else None,
        "null_reason": None if good else "required seed/frame/policy failed or missing"}


def values_from_cases(cases):
    return {metric: {name: case["metrics"][metric] if case["status"] == "complete" else None
                    for name, case in cases.items()} for metric in evaluate.METRICS}


def summary_envelope(values):
    return {metric: {control: evaluate.envelope([policies[name] for name in evaluate.RANDOM_CASES], policies[control])
                    for control in evaluate.CONTROLS} for metric, policies in values.items()}


def guard_output(output, n):
    try:
        bridge.validate_output(output, n)
        return None
    except full.RolloutGuard as error:
        return error.details


def audit_seconds(value, audit, message, allow_null=False):
    audit.check((allow_null and value is None) or (type(value) in (int,float) and np.isfinite(value) and value>=0),message)


def check_parity(row, a, base, audit):
    saved = row["native_parity"]
    native_values = {key:a[f"native_{key}"] for key in ("prediction", "raw_risk", "risk")}
    supplied = {key:a[f"supplied_parity_{key}"] for key in native_values}
    finite = all(np.isfinite(value).all() for values in (native_values,supplied) for value in values.values())
    features = [np.array_equal(a["native_node_features"], a["supplied_node_features"]),
                np.array_equal(a["native_edges"], base),
                np.array_equal(a["native_edge_features"], a["supplied_edge_features"])]
    agree = bool(finite and np.allclose(native_values["prediction"], supplied["prediction"],rtol=0,atol=bridge.PARITY_PRED_ATOL)
                 and all(np.allclose(native_values[key],supplied[key],rtol=bridge.PARITY_RISK_RTOL,atol=bridge.PARITY_RISK_ATOL)
                         for key in ("raw_risk","risk")))
    fields = {"native_edge_identity":features[1], "native_supplied_feature_identity":features,
        "finite":finite, "prediction_risk_agree":agree, "passed":bool(all(features) and agree),
        "prediction_atol":bridge.PARITY_PRED_ATOL,"risk_atol":bridge.PARITY_RISK_ATOL,"risk_rtol":bridge.PARITY_RISK_RTOL}
    fields.update({key+"_max_abs_difference":float(np.max(np.abs(native_values[key]-supplied[key]))) if finite else None
                   for key in native_values})
    audit.equal(saved,fields,"Native parity scalar/flags differ from arrays")
    return supplied


def audit_frame(row, a, audit):
    """Rebuild each graph/selection and scalar without calling a network."""
    audit.equal(set(row["cases"]),set(evaluate.POLICIES),"Exact13 cases required")
    audit_seconds(row["operational_seconds"],audit,"Bad frame operational time")
    audit.check(row["status"] in ("complete","failed"),"Unknown frame status")
    audit.equal(row["status"]=="complete",row["failure"] is None,"Frame failure consistency")
    for case in row["cases"].values():
        audit.check(case["status"] in ("complete","failed"),"Unknown case status")
    for history in ("current_history","previous_history"):
        audit.equal(full.state_hash(a[history]),row[history+"_sha256"],"Observed history hash differs")
    cur,prev,types=a["current_history"],a["previous_history"],a["particle_types"]
    n=row["n_particles"]
    audit.check(cur.shape==prev.shape==(6,n,2) and types.shape==(n,) and not np.any(types==3),"Input shapes/types differ")
    std,bounds=a["acceleration_std"],a["bounds"]
    inputs={key:full.state_hash(a[key]) for key in
            ("current_history","previous_history","particle_types","acceleration_std","bounds")}
    audit.check(std.shape==(2,) and np.isfinite(std).all() and np.all(std>0),"Bad normalization")
    audit.check(bounds.shape==(2,2) and np.isfinite(bounds).all() and np.all(bounds[:,1]>bounds[:,0]),"Bad bounds")
    audit.equal(row["random_seed_materials"],{name:evaluate.random_material(row["seed"],row,draw)
                 for draw,name in enumerate(evaluate.RANDOM_CASES)},"Random stream material differs")
    try:
        full.check_state(cur,"current_history",bridge.MAX_ABS)
        graph,base,_,base_audit=native.native_graph(cur,"base",None,np.random.default_rng(0))
    except full.RolloutGuard as error:
        phase="current_history" if error.details.get("category")!="candidate_pair_resource_guard" else "current_graph"
        audit.equal(row["failure"],{**error.details,"phase":phase},"Current guard failure differs")
        audit.check(all(case["status"]=="failed" and case["metrics"] is None for case in row["cases"].values()),"Guard must leave13 null cases")
        for case in row["cases"].values():
            audit.equal(case["failure"],row["failure"],"Current guard case cause differs")
        audit.equal(row["random_envelope"],evaluate.frame_envelopes(row["cases"]),"Guard envelope differs")
        return {**{k:row[k] for k in IDENTITY},"values":values_from_cases(row["cases"]),"envelope":row["random_envelope"],
                "failure":row["failure"],"case_status":{name:case["status"] for name,case in row["cases"].items()},
                "operational_seconds":row["operational_seconds"],"input_identity":inputs}
    audit.array(a["strict_base_pairs"],graph.base,"Geometric base differs")
    audit.array(a["strict_annulus_pairs"],graph.extra,"Geometric annulus differs")
    audit.equal(row["optional_budget"],int(.25*len(graph.extra)),"Optional budget differs")
    base_output=check_parity(row,a,base,audit)
    audit.check(row["native_parity"]["passed"],"Parity failure cannot enter complete scientific summary")
    base_failure=guard_output(base_output,n)
    previous_output,previous_failure=None,None
    phase="previous_history"
    try:
        full.check_state(prev,phase,bridge.MAX_ABS)
        phase="previous_graph"
        _,old,_,old_audit=native.native_graph(prev,"base",None,np.random.default_rng(0))
        audit.array(a["previous_base_edges"],old,"Previous native graph differs")
        previous_output={key:a[f"previous_base_{key}"] for key in ("prediction","raw_risk","risk")}
        phase="previous_forward"
        previous_failure=guard_output(previous_output,n)
        if previous_failure: previous_failure={**previous_failure,"phase":phase}
        else: audit.equal(row["previous_score"]["graph"],old_audit,"Previous scoring graph audit differs")
    except full.RolloutGuard as error:
        previous_failure={**error.details,"phase":phase}
    audit.equal(row["previous_score"]["failure"],previous_failure,"Previous scoring failure differs")
    audit.equal(row["previous_score"]["status"],"failed" if previous_failure else "complete","Previous scoring status differs")
    audit_seconds(row["previous_score"]["operational_seconds"],audit,"Bad previous scoring time",allow_null=previous_output is None)
    target=a["target_position"]
    audit.check(target.shape==(n,2) and np.isfinite(target).all(),"Target shape/finiteness differs")
    audit.equal(full.state_hash(target),row["target_sha256"],"Target hash differs")
    inputs["target_position"]=full.state_hash(target)
    audit.equal(row["ground_truth_boundary"],full.boundary_metrics(target,bounds),"Truth boundary differs")
    audit.equal(row["current_observed_boundary"],full.boundary_metrics(cur[-1],bounds),"Observed boundary differs")
    ses={}
    for name in evaluate.POLICIES:
        case=row["cases"][name]
        audit.equal(case["status"]=="complete",case["failure"] is None,"Case failure consistency")
        scoring_failure=previous_failure if name==evaluate.RISKS[0] else base_failure if name==evaluate.RISKS[1] else None
        if scoring_failure:
            audit.equal(case["failure"],{"category":"scoring_pass_failed","cause":scoring_failure},"Failed scoring cause differs")
            audit.check(case["metrics"] is None and f"prediction__{name}" not in a,"Scoring failure has output/metric")
            audit.equal(case["network_passes_for_standalone_policy"],
                (1 if previous_output is not None else 0) if name==evaluate.RISKS[0] else 1,
                "Failed scoring attempted pass count differs")
            continue
        actual,score=evaluate.selection_spec(name,previous_output,base_output)
        rng=np.random.default_rng(np.random.SeedSequence(row["random_seed_materials"].get(name,[20261006,881,0])))
        selected=full.choose_pairs(graph,actual,cur,score,rng)
        optional=bridge.same.canonical_pairs(selected[len(graph.base):])
        edges=np.concatenate((base,bridge.ordered_edges(cur[-1],optional,False)),axis=1)
        audit.array(a[f"optional_pairs__{name}"],optional,"Optional draw/risk/speed mask differs")
        audit.array(a[f"edges__{name}"],edges,"Native prefix/optional suffix differs")
        expected_graph={**base_audit,"retained_optional_pairs":len(optional),"directed_edges":edges.shape[1],
            "selected_optional_pair_sha256":full.state_hash(optional),"directed_edge_sha256":full.state_hash(edges)}
        audit.equal(case["graph"],expected_graph,"Case graph hash/count/prefix audit differs")
        if score is not None: audit.array(a[f"selection_score__{name}"],score,"Risk source differs")
        if name=="speed25": audit.array(a[f"selection_score__{name}"],np.linalg.norm(cur[-1]-cur[-2],axis=-1),"Speed source differs")
        output={key:a[f"{key}__{name}"] for key in ("prediction","raw_risk","risk")}
        failure=guard_output(output,n)
        audit.equal(case["failure"],{**failure,"phase":"forward"} if failure else None,"Output guard/status differs")
        if name=="base":
            for key in output:audit.array(output[key],base_output[key],"Base must reuse exact parity output")
        audit.equal(case["reused_native_parity_output"],name=="base","Parity reuse flag differs")
        audit.equal(case["network_passes_for_standalone_policy"],2 if name in evaluate.RISKS else 1,"Standalone pass count differs")
        for key in ("graph_build_and_selection_seconds","forward_operational_seconds","scoring_seconds"):
            audit.check(type(case[key]) in (float,int) and np.isfinite(case[key]) and case[key]>=0,"Bad timing")
        expected_score_seconds=(row["previous_score"]["operational_seconds"] if name==evaluate.RISKS[0]
                                else row["cases"]["base"]["forward_operational_seconds"] if name==evaluate.RISKS[1] else 0.)
        audit.equal(case["scoring_seconds"],expected_score_seconds,"Cached scoring time identity differs")
        if np.isfinite(output["prediction"]).all():
            audit.equal(case["predicted_boundary"],full.boundary_metrics(output["prediction"],bounds),"Prediction boundary differs")
        if failure:
            audit.check(case["metrics"] is None,"Failed action has metric")
            continue
        metrics,residual,normalized=bridge.residual_record(output,target,std)
        audit.equal(case["metrics"],metrics,"Coordinate MSE differs")
        audit.array(a[f"position_residual__{name}"],residual,"Physical residual differs")
        audit.array(a[f"normalized_residual__{name}"],normalized,"Normalized residual differs")
        ses[name]=np.mean(normalized**2,axis=-1)
        audit.array(a[f"normalized_coordinate_se__{name}"],ses[name],"Particle SE differs")
    for name in evaluate.POLICIES:
        benefit=row["benefits"][name]
        if "base" not in ses or name not in ses:
            audit.check(benefit["mean_signed_normalized_coordinate_gain"] is None and benefit.get("reason"),"Failed benefit lacks null reason")
            continue
        gain=ses["base"]-ses[name]
        audit.array(a[f"signed_normalized_coordinate_gain__{name}"],gain,"Signed action gain differs")
        audit.scalar(benefit["mean_signed_normalized_coordinate_gain"],float(gain.mean()),"Mean gain differs")
        audit.scalar(benefit["harmful_particle_fraction"],float(np.mean(gain<0)),"Harm fraction differs")
        if previous_failure is None:
            audit.equal(row["previous_risk_correlations"][name],bridge.same.spearman(previous_output["risk"],gain),"Previous-risk gain ranks differ")
    corr=bridge.same.spearman(previous_output["risk"],ses["base"]) if previous_failure is None and "base" in ses else {"value":None,"reason":"required scoring or base output failed"}
    audit.equal(row["previous_risk_base_residual_correlation"],corr,"Previous-risk residual ranks differ")
    failures=[{"case":name,"failure":case["failure"]} for name,case in row["cases"].items() if case["status"]=="failed"]
    audit.equal(row["failure"],{"category":"case_failures","cases":failures} if failures else None,"Frame failure ledger differs")
    audit.equal(row["random_envelope"],evaluate.frame_envelopes(row["cases"]),"Saved frame envelope differs")
    return {**{key:row[key] for key in IDENTITY},"values":values_from_cases(row["cases"]),
        "input_identity":inputs,
        "envelope":row["random_envelope"],"failure":row["failure"],
        "case_status":{name:case["status"] for name,case in row["cases"].items()},
        "operational_seconds":row["operational_seconds"],
        "benefits":row["benefits"],"previous_risk_correlations":row["previous_risk_correlations"],
        "ground_truth_boundary":row["ground_truth_boundary"],"current_observed_boundary":row["current_observed_boundary"],
        "case_graphs":{name:case.get("graph") for name,case in row["cases"].items()},
        "case_boundaries":{name:case.get("predicted_boundary") for name,case in row["cases"].items()},
        "case_timings":{name:{key:case.get(key) for key in ("graph_build_and_selection_seconds","forward_operational_seconds","scoring_seconds","network_passes_for_standalone_policy")} for name,case in row["cases"].items()}}


def combine_values(rows):
    return {metric:{name:complete_mean([row["values"][metric][name] for row in rows])
                    for name in evaluate.POLICIES} for metric in evaluate.METRICS}


def summarize_rows(rows, expected):
    """Missing units have null values, exact hierarchy, envelope at each level."""
    wanted={tuple(item[key] for key in IDENTITY):item for item in expected}
    evaluate.require(len(wanted)==len(expected) and expected,"Unique nonempty expected units required")
    found={}
    for row in rows:
        key=tuple(row[k] for k in IDENTITY)
        evaluate.require(key in wanted and key not in found,"Unknown/duplicate observed unit")
        evaluate.require(set(row["values"])==set(evaluate.METRICS) and all(set(x)==set(evaluate.POLICIES) for x in row["values"].values()),"Exact metric/case coverage required")
        found[key]=row
    grouped=defaultdict(list)
    for key,item in wanted.items():
        blank={**item,"values":{metric:{name:None for name in evaluate.POLICIES} for metric in evaluate.METRICS},"failure":{"category":"missing_required_frame"}}
        chosen=found.get(key,blank)
        grouped[item["trajectory_id"]].append({**chosen,"_summary_envelope":summary_envelope(chosen["values"])})
    trajectories={}
    for trajectory,items in sorted(grouped.items()):
        values=combine_values(items)
        trajectories[trajectory]={"values":values,"envelope_of_trajectory_mean_errors":summary_envelope(values),
            "required_frames":len(items),"missing_frames":sum(tuple(r[k] for k in IDENTITY) not in found for r in items),
            "defined_frames":{metric:{name:sum(row["values"][metric][name] is not None for row in items) for name in evaluate.POLICIES} for metric in evaluate.METRICS}}
    seed_values=combine_values(list(trajectories.values()))
    # Average frame-level contrasts/counts after within-trajectory reduction;
    # these differ from rank/envelope of the trajectory/seed mean errors.
    frame_measures={}
    for metric in evaluate.METRICS:
        frame_measures[metric]={}
        for control in evaluate.CONTROLS:
            by_field={}
            for field in ("reference_minus_random_mean","random_strictly_better","random_tied","random_strictly_worse","minimum","maximum","sample_sd"):
                values=[]
                for items in grouped.values():
                    values.append(complete_mean([row["_summary_envelope"][metric][control][field] for row in items]))
                by_field[field]=complete_mean(values)
            frame_measures[metric][control]=by_field
    return {"values":seed_values,"envelope_of_equal_trajectory_seed_mean_errors":summary_envelope(seed_values),
        "equal_trajectory_mean_of_frame_envelope_measures":frame_measures,"trajectories":trajectories,
        "required_frames":len(expected),"present_frames":len(rows),"missing_frames":len(expected)-len(rows),
        "required_trajectories":len(grouped),"frames":rows}


def summarize_cohort(runs):
    evaluate.require(all(type(row["seed"]) is int for row in runs),"Nonboolean integer seed required")
    lookup={(row["objective"],row["seed"]):row for row in runs}
    evaluate.require(len(runs)==6 and set(lookup)==evaluate.COHORT,"Exactly six unique faithful/NLL original seeds required")
    result={}
    for split in ("valid","test"):
        result[split]={}
        for objective in ("faithful","nll"):
            seeds=[lookup[objective,seed]["splits"][split] for seed in (0,1,2)]
            result[split][objective]={"seeds":{str(seed):seeds[seed] for seed in (0,1,2)},
                "case_error_statistics":{metric:{name:seed_statistics([seed["values"][metric][name] for seed in seeds])
                    for name in evaluate.POLICIES} for metric in evaluate.METRICS},
                "control_minus_random_mean_statistics":{metric:{control:seed_statistics([seed["equal_trajectory_mean_of_frame_envelope_measures"][metric][control]["reference_minus_random_mean"] for seed in seeds])
                    for control in evaluate.CONTROLS} for metric in evaluate.METRICS},
                "mean_frame_comparison_statistics":{metric:{control:{field:seed_statistics([seed["equal_trajectory_mean_of_frame_envelope_measures"][metric][control][field] for seed in seeds])
                    for field in ("reference_minus_random_mean","random_strictly_better","random_tied","random_strictly_worse","minimum","maximum","sample_sd")} for control in evaluate.CONTROLS} for metric in evaluate.METRICS}}
    return result


def load_run(directory, audit, cache):
    directory=Path(directory).resolve()
    audit.check(not (directory/"run.lock").exists() and not list(directory.rglob("*.tmp")),"Live lock or unresolved temporary files")
    protocol,result,status=(evaluate.strict_json(directory/name) for name in ("protocol.json","result.json","status.json"))
    ph,rh=full.sha256(directory/"protocol.json"),full.sha256(directory/"result.json")
    for key,expected_value in (("scope",evaluate.SCOPE),("schema",1),("cases",list(evaluate.POLICIES)),
                               ("random_draw_ids",list(range(8)))):
        audit.equal(protocol[key],expected_value,"Envelope scope/schema/cases/draws differ")
    objective,seed=protocol["objective"],protocol["seed"]
    audit.check(type(seed) is int and (objective,seed) in evaluate.COHORT,"Unknown original objective/seed")
    audit.check(status["state"]==result["state"]=="complete" and status["result_sha256"]==rh
                and result["protocol_sha256"]==ph,"Uncommitted/incomplete envelope result")
    audit.equal(result["scope"],evaluate.SCOPE,"Result scope differs")
    audit_seconds(result["operational_seconds"],audit,"Bad result operational time")
    audit_seconds(status["operational_seconds"],audit,"Bad status operational time")
    for key in ("objective","seed","checkpoint_sha256"):
        audit.equal(result[key],protocol[key],"Result identity differs")
    for key in ("objective","seed"):
        audit.equal(status[key],protocol[key],"Status identity differs")
    evaluate.verify_files(protocol["input_files_sha256"])
    for path in evaluate.source_paths():
        audit.equal(protocol["input_files_sha256"].get(str(path.resolve())),full.sha256(path),"Required source/protocol pin absent")
    checkpoint=Path(protocol["checkpoint"])
    audit.equal(full.sha256(checkpoint),protocol["checkpoint_sha256"],"Model byte identity differs")
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    valid_path,test_path=Path(protocol["validation_manifest"]),Path(protocol["test_manifest"])
    metadata=evaluate.strict_json(valid_path.parent/"metadata.json")
    full.check_checkpoint(payload,"locked_test",evaluate.TRAINING_PROTOCOL,full.sha256(valid_path.parent/"metadata.json"))
    evaluate.committed_checkpoint(checkpoint,payload)
    expected,valid,test=bridge.expected_frames(payload["run_config"],valid_path,test_path)
    audit.equal(protocol["expected_frames"],expected,"Original fixed425 schedule differs")
    audit.equal(protocol["run_config_sha256"],payload["run_config_sha256"],"Original run config differs")
    audit.check(payload["run_config"]["objective"]==objective and payload["run_config"]["seed"]==seed,"Checkpoint objective/seed differs")
    normalization=payload["simulator_config"]["normalization_stats"]["acceleration"]["std"].detach().cpu().numpy().astype(np.float64)
    saved_dir=Path(protocol["saved_same_state_dir"])
    lookup,saved=bridge.verify_saved_test(saved_dir,protocol["checkpoint_sha256"],expected)
    audit.equal(protocol["saved_test_inputs"],saved,"Saved original history admission differs")
    audit.equal(protocol["source_tfrecord"],test["source"],"Official test source differs")
    audit.check(valid["metadata"]==test["metadata"]==metadata,"Split metadata differs")
    verify_required_input_pins(protocol,payload["run_config"],valid,test,lookup,audit)
    trajectories={}
    for split,path in (("valid",valid_path),("test",test_path)):
        identity=(str(path),full.sha256(path))
        if identity not in cache:cache[identity]=full.load_manifest_data(path,verify_hashes=True)
        trajectories[split]=cache[identity]
    audit.equal(result["required_frames"],425,"425 required frames")
    audit.equal(len(result["records"]),425,"425 committed frame records required")
    expected_lookup={bridge.stem(item):item for item in expected}
    seen=set();rows=[];hashes={str(directory/name):full.sha256(directory/name) for name in ("protocol.json","result.json","status.json")}
    for index in result["records"]:
        name=index["record_file"]
        audit.check(Path(name).name==name and Path(name).stem in expected_lookup and name not in seen,"Unknown/duplicate/unsafe frame record")
        seen.add(name);path=directory/name
        audit.equal(full.sha256(path),index["record_sha256"],"Frame JSON hash differs")
        row=evaluate.strict_json(path)
        audit.equal(bridge.record_index(row,path),index,"Compact frame index differs")
        item=expected_lookup[Path(name).stem]
        for key,value in item.items():audit.equal(row[key],value,"Fixed frame identity differs")
        for key,value in (("objective",objective),("seed",seed),("checkpoint_sha256",protocol["checkpoint_sha256"]),("protocol_sha256",ph)):
            audit.equal(row[key],value,"Frame model/protocol identity differs")
        array=directory/(Path(name).stem+".npz")
        audit.equal(row["array_file"],array.name,"Array path differs")
        audit.equal(full.sha256(array),row["array_sha256"],"Array hash differs")
        with np.load(array,allow_pickle=False) as arrays:
            source=bridge.frame_input(item,trajectories,saved_dir,lookup)
            for key,value in zip(("current_history","previous_history","particle_types","target_position"),source):
                if key in arrays:audit.array(arrays[key],value,"Source history/target/type correspondence differs")
            audit.array(arrays["bounds"],np.asarray(metadata["bounds"]),"Metadata bounds differ")
            audit.array(arrays["acceleration_std"],normalization,"Checkpoint acceleration normalization differs")
            rows.append(audit_frame(row,arrays,audit))
        hashes[str(path)],hashes[str(array)]=index["record_sha256"],row["array_sha256"]
    expected_names={bridge.stem(item)+suffix for item in expected for suffix in (".json",".npz")}|{"protocol.json","result.json","status.json"}
    audit.equal({p.name for p in directory.iterdir() if p.suffix in (".json",".npz")},expected_names,"Unexpected raw result files")
    audit.equal(result["complete_frames"],sum(row["failure"] is None for row in rows),"Complete frame count differs")
    audit.equal(result["failed_frames"],sum(row["failure"] is not None for row in rows),"Failed frame count differs")
    audit.equal(status["committed_frames"],425,"Committed count differs")
    evaluate.verify_files(protocol["input_files_sha256"]);evaluate.verify_files(hashes)
    return {"objective":objective,"seed":seed,"checkpoint_sha256":protocol["checkpoint_sha256"],
        "protocol_sha256":ph,"raw_files_sha256":hashes,"input_files_sha256":protocol["input_files_sha256"],
        "required_frames":425,"complete_frames":result["complete_frames"],"failed_frames":result["failed_frames"],
        "operational_seconds":result["operational_seconds"],
        "splits":{split:summarize_rows([row for row in rows if row["split"]==split],[item for item in expected if item["split"]==split]) for split in ("valid","test")}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",nargs=6,type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    evaluate.require(len({path.resolve() for path in args.runs})==6,"Six distinct directories required")
    preflight_summary_output(args.output_dir,args.runs)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();audit=Audit();cache={}
    pins={str(path.resolve()):full.sha256(path) for path in [Path(__file__),*evaluate.source_paths()]}
    full.atomic_json(args.output_dir/"analysis_source.json",{"source_files_sha256":pins,"scope":evaluate.SCOPE})
    try:
        runs=[load_run(path,audit,cache) for path in args.runs]
        audit.check(len({row["checkpoint_sha256"] for row in runs})==6,"Six distinct original models required")
        for split in ("valid","test"):
            ref=runs[0]["splits"][split]
            for run in runs[1:]:
                audit.equal([tuple(row[k] for k in IDENTITY) for row in run["splits"][split]["frames"]],
                            [tuple(row[k] for k in IDENTITY) for row in ref["frames"]],"Paired histories differ")
                audit.equal([row["input_identity"] for row in run["splits"][split]["frames"]],
                            [row["input_identity"] for row in ref["frames"]],"Paired input bytes/normalization differ")
        result={"scope":evaluate.SCOPE,"runs":runs,"summary":summarize_cohort(runs),"checks":audit.checks,
            "coverage":{"models":6,"frames":2550,"case_slots":33150,"random_slots":20400},
            "analysis_seconds":time.perf_counter()-started,"source_files_sha256":pins,
            "limits":"Observed-history conditional Monte Carlo robustness; no extra model replication, independence, significance, causal concentration, autonomous or speedup claim."}
        reverify_cohort(runs,pins)
        full.atomic_json(args.output_dir/"results.json",result)
        full.atomic_json(args.output_dir/"status.json",{"state":"complete","result_sha256":full.sha256(args.output_dir/"results.json"),"checks":audit.checks,"analysis_seconds":time.perf_counter()-started})
    except BaseException as error:
        full.atomic_json(args.output_dir/"status.json",{"state":"error","error_type":type(error).__name__,"error":str(error),"checks":audit.checks})
        raise


def preflight_summary_output(output_dir, run_dirs):
    """Inspect only metadata before any write; protect every admitted input tree."""
    protected=list(run_dirs)+[path.parent for path in evaluate.source_paths()]+[Path(__file__).parent]
    for run_dir in run_dirs:
        protocol=evaluate.strict_json(Path(run_dir)/"protocol.json")
        for key in ("checkpoint","validation_manifest","test_manifest"):
            path=Path(protocol[key])
            evaluate.require(path.is_absolute(),"Absolute admitted input path required")
            protected.append(path.resolve().parent)
        protected.append(Path(protocol["saved_same_state_dir"]).resolve())
        config=evaluate.strict_json(Path(protocol["checkpoint"]).parent/"protocol.json")
        valid=evaluate.strict_json(protocol["validation_manifest"])
        test=evaluate.strict_json(protocol["test_manifest"])
        saved=evaluate.strict_json(Path(protocol["saved_same_state_dir"])/"result.json")
        lookup={i:row for i,row in enumerate(saved["records"])}
        protected += [Path(name).parent for name in required_input_paths(protocol,config,valid,test,lookup)]
        for name in protocol["input_files_sha256"]:
            path=Path(name)
            evaluate.require(path.is_absolute(),"Absolute pinned input path required")
            protected.append(path.resolve().parent)
    return evaluate.preflight_output(output_dir,protected)


def required_input_paths(protocol, run_config, valid, test, saved_lookup):
    """Reconstruct producer admission paths independently of its supplied pin map."""
    checkpoint=Path(protocol["checkpoint"])
    valid_path,test_path=Path(protocol["validation_manifest"]),Path(protocol["test_manifest"])
    saved=Path(protocol["saved_same_state_dir"])
    paths=evaluate.source_paths()+[checkpoint,valid_path,test_path,valid_path.parent/"metadata.json",
                                  saved/"protocol.json",saved/"result.json"]
    paths += [checkpoint.parent/name for name in ("protocol.json","latest.json","status.json")]
    paths += [full.REPO/relative for relative in run_config["source_sha256"]]
    paths += [saved/row[key] for row in saved_lookup.values() for key in ("array_file","record_file")]
    paths += [directory/row[key]["path"] for manifest,directory in ((valid,valid_path.parent),(test,test_path.parent))
              for row in manifest["records"] for key in ("positions","particle_types")]
    return {str(path.resolve()) for path in paths}


def verify_required_input_pins(protocol, run_config, valid, test, saved_lookup, audit):
    required=required_input_paths(protocol,run_config,valid,test,saved_lookup)
    audit.equal(set(protocol["input_files_sha256"]),required,"Required input pin key coverage differs")
    for name in required:
        audit.equal(protocol["input_files_sha256"][name],full.sha256(name),"Required input byte hash differs")


def reverify_cohort(runs, source_pins):
    """Repeat every prior input/raw identity immediately before publication."""
    for run in runs:
        evaluate.verify_files(run["raw_files_sha256"])
        evaluate.verify_files(run["input_files_sha256"])
    evaluate.verify_files(source_pins)


if __name__=="__main__":main()
