#!/usr/bin/env python3
"""Independently check published Goop paired scalars against passed array audits.

Reads two small audit receipts and a scalar summary. No model, dataset, remote
host, or frozen evaluation/scalar module is imported. Null values never drop
out of fixed three-seed means, sample SDs, or contrasts.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import traceback

import audit_goop_saved_arrays_v1 as saved


def contrast(a,b): return a-b if saved.finite(a) and saved.finite(b) else None


def seed_stats(values):
    if len(values)!=3: raise ValueError("Exactly three ordered seed values required")
    complete=all(saved.finite(v) for v in values)
    return {"seed_values":{str(s):v for s,v in enumerate(values)},"required_seed_pairs":3,
            "defined_seed_pairs":sum(saved.finite(v) for v in values),
            "mean":saved.mean(values),"sample_sd":statistics.stdev(values) if complete else None}


def recompute(audits):
    """Return only independently supported fields, retaining all fixed cells."""
    checks=saved.Checks();grid={}
    for role,audit in audits.items():
        checks.equal(audit["host_role"],role,"audit role")
        for model in audit["models"]:
            key=model["arm"],model["seed"],model["stage"]
            checks.require(key not in grid,"unique model-stage")
            grid[key]=model
    checks.equal(set(grid),{(a,s,t) for a in ("base","mix") for s in range(3) for t in saved.STAGES},"complete six-model/stage grid")
    full={"absolute":{},"within_arm_policy_contrasts":{},"mix_minus_base":{},"risk_minus_random_mix_minus_base_interaction":{}}
    for metric in saved.FULL_METRICS:
        def value(a,s,p): return grid[a,s,"full_rollout_test"]["metrics"][metric+"/"+p]
        full["absolute"][metric]={a:{p:seed_stats([value(a,s,p) for s in range(3)]) for p in saved.POLICIES} for a in ("base","mix")}
        full["within_arm_policy_contrasts"][metric]={a:{p+"_minus_"+ref:seed_stats([contrast(value(a,s,p),value(a,s,ref)) for s in range(3)])
            for ref in ("base","random25") for p in saved.POLICIES if p!=ref} for a in ("base","mix")}
        full["mix_minus_base"][metric]={p:seed_stats([contrast(value("mix",s,p),value("base",s,p)) for s in range(3)]) for p in saved.POLICIES}
        full["risk_minus_random_mix_minus_base_interaction"][metric]=seed_stats([
            contrast(contrast(value("mix",s,"laggedrisk25"),value("mix",s,"random25")),
                     contrast(value("base",s,"laggedrisk25"),value("base",s,"random25"))) for s in range(3)])
    counts=Counter(cell["state"] for key,m in grid.items() if key[2]=="full_rollout_test" for cell in m["cells"])
    checks.equal(sum(counts.values()),1080,"1080 required rollout cells")
    full.update(required_outcomes=1080,coverage=dict(counts),all_required_outcomes_recorded=counts["completed_required_outcome"]+counts["recorded_failed_outcome"]==1080,
                all_required_outcomes_complete=counts["completed_required_outcome"]==1080)
    full["coverage_by_model"]=[]
    full["failed_accepted_prefixes"]=[]
    for audit in audits.values():
        for model in audit["models"]:
            if model["stage"]!="full_rollout_test": continue
            full["coverage_by_model"].append({"arm":model["arm"],"seed":model["seed"],
                "counts":dict(Counter(c["state"] for c in model["cells"])),
                "guard_or_execution_failure_categories":dict(Counter(c["failure"].get("category","unspecified") for c in model["cells"] if c["state"]=="recorded_failed_outcome"))})
        for row in audit["row_checks"]:
            if row["stage"]!="full_rollout_test" or row["status"]!="failed": continue
            full["failed_accepted_prefixes"].append({"arm":row["arm"],"seed":row["seed"],"source_index":row["unit"][0],"policy":row["unit"][1],
                "completed_steps":row["completed_steps"],"failure":row["failure"],"accepted_prefix_boundary":row["accepted_prefix_boundary"]})
    diagnostics={}
    for stage in saved.STAGES[1:]:
        keys=set(grid["base",0,stage]["metrics"])
        for a in ("base","mix"):
            for s in range(3): checks.equal(set(grid[a,s,stage]["metrics"]),keys,"diagnostic leaf grid")
        def value(a,s,k): return grid[a,s,stage]["metrics"][k]
        item={"absolute":{a:{k:seed_stats([value(a,s,k) for s in range(3)]) for k in sorted(keys)} for a in ("base","mix")},
              "mix_minus_base":{k:seed_stats([contrast(value("mix",s,k),value("base",s,k)) for s in range(3)]) for k in sorted(keys)},
              "coverage":{f"{a}_seed{s}":dict(Counter(c["state"] for c in grid[a,s,stage]["cells"])) for a in ("base","mix") for s in range(3)}}
        if stage!="clean_validation":
            risk="accuracy/previous-observed-base-risk25/position_coordinate_mse";random="accuracy/random25/position_coordinate_mse"
            within={a:[contrast(value(a,s,risk),value(a,s,random)) for s in range(3)] for a in ("base","mix")}
            item["previous_observed_risk_minus_random_position_mse"]={a:seed_stats(v) for a,v in within.items()}
            item["risk_minus_random_mix_minus_base_interaction"]=seed_stats([contrast(within["mix"][s],within["base"][s]) for s in range(3)])
        diagnostics[stage]=item
    return {"full_rollout":full,"diagnostics":diagnostics}


def supported_equal(actual,expected,checks,label="summary"):
    """Published summary may contain extra operational fields; check all ours."""
    extensible={"summary","summary/full_rollout"} | {"summary/diagnostics/"+s for s in saved.STAGES[1:]}
    if label not in extensible: checks.equal(set(actual),set(expected),label+" exact arithmetic/coverage keys")
    for key,value in expected.items():
        checks.require(key in actual,label+" missing "+key)
        if isinstance(value,dict): supported_equal(actual[key],value,checks,label+"/"+key)
        else: checks.close(actual[key],value,label+"/"+key)


def audit_summary(audit_paths,audit_hashes,summary_path,summary_sha):
    checks=saved.Checks();audits={}
    source_paths=[Path(__file__).resolve(),Path(saved.__file__).resolve(),saved.HERE/"goop_saved_diagnostic_audit_v1.py"]
    bindings={str(p):saved.digest(p.read_bytes()) for p in source_paths}
    shared={}
    for role in ("A","B"):
        path=audit_paths[role];audit=saved.load_json(path,audit_hashes[role]);bindings[str(path)]=audit_hashes[role]
        checks.equal((audit["schema"],audit["status"],audit["host_role"]),("goop_saved_array_audit_v1","passed_supported_checks",role),"passed array audit identity")
        checks.equal(audit["audit_revision"],2,"array audit failure-accounting revision")
        checks.equal(audit["auditor_sha256"],bindings[str(source_paths[1])],"same current saved-array auditor")
        checks.equal(audit["diagnostic_helper_sha256"],bindings[str(source_paths[2])],"same current diagnostic helper")
        for key,value in audit["shared_saved_state_hashes"].items():
            if key in shared: checks.equal(value,shared[key],"cross-host saved observed state/target identity")
            shared[key]=value
        audits[role]=audit
    summary=saved.load_json(summary_path,summary_sha);bindings[str(summary_path)]=summary_sha
    checks.equal((summary["schema"],summary["status"]),("adaptgns_goop_graph_support_paired_scalar_summary_scoped_v3","fixed_scalar_aggregation_complete"),"published summary identity")
    checks.equal(summary["summarizer_sha256"],saved.COLLECTOR_SHA,"summary reviewed source")
    checks.equal(summary["source_sha256"],saved.SOURCE_PINS,"summary frozen imports")
    checks.equal(summary["protocol_sha256"],saved.PROTOCOL_SHA,"summary frozen protocol")
    for role,audit in audits.items():
        checks.equal(audit["cohort_sha256"],summary["cohort_sha256"],"same frozen cohort")
        checks.equal(audit["collection_sha256"],summary["collection_sha256"][role],"same stopped collection")
    computed=recompute(audits);supported_equal(summary,computed,checks)
    for path,expected in bindings.items(): checks.equal(saved.file_hash(path),expected,"bound input/source unchanged")
    return {"schema":"goop_paired_saved_array_audit_v1","audit_revision":2,"status":"passed_supported_checks","checks":checks.count,
            "cohort_sha256":summary["cohort_sha256"],"summary_sha256":summary_sha,"verified_input_sha256":bindings,
            "verified_paired_scalars":computed,"scope":"Same saved-array/scalar-series scope as input audits; all fixed seed means/sampleSD/contrasts and null propagation independently recomputed. Operational runtime and external source/model truth not remeasured."}


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--execute",action="store_true")
    for name in ("audit-a","audit-b","summary","output"): p.add_argument("--"+name,type=Path)
    for name in ("audit-a","audit-b","summary"): p.add_argument("--"+name+"-sha256")
    args=p.parse_args()
    if not args.execute: print(json.dumps({"status":"description_only","scope":__doc__}));return 0
    if not all(vars(args).values()): p.error("all exact SHA-bound inputs and fresh output required")
    if args.output.exists(): p.error("output must be fresh")
    try:
        result=audit_summary({"A":args.audit_a,"B":args.audit_b},{"A":args.audit_a_sha256,"B":args.audit_b_sha256},args.summary,args.summary_sha256)
        encoded=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
        for path,expected in result["verified_input_sha256"].items():
            if saved.file_hash(path)!=expected: raise ValueError("Input/source changed before publication: "+path)
    except Exception as error:
        failure={"schema":"goop_paired_saved_array_audit_v1","status":"failed","error":str(error),"traceback":traceback.format_exc()}
        with args.output.open("x") as stream: json.dump(failure,stream,indent=2,sort_keys=True,allow_nan=False)
        raise
    with args.output.open("xb") as stream: stream.write(encoded)
    print(json.dumps({"status":result["status"],"checks":result["checks"]}));return 0


if __name__=="__main__": raise SystemExit(main())
