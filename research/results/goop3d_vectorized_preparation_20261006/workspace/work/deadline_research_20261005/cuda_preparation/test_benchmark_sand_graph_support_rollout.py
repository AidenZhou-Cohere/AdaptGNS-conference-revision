"""Synthetic checkpoint dictionaries and mocked rollouts; no model/data access."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parent


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+".py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


bench=load("benchmark_sand_graph_support_rollout")
trainer=load("train_sand_graph_support_cuda")


def checkpoint_fixture(tmp_path):
    args=SimpleNamespace(objective="faithful",arm="mix",seed=0,checkpoint_updates=512,
        trainer_source=ROOT/"train_sand_graph_support_cuda.py",train_admission=tmp_path/"train_admission.json",
        training_protocol=tmp_path/"training_protocol.md")
    args.training_protocol.write_text("synthetic timing protocol")
    admission={"schema":"adaptgns_sand_training_admission_v1","status":"admitted","dataset":"Sand",
        "manifest_sha256":"a"*64,"frames_per_trajectory":320,"particle_type_ids":[6],
        "converter_sha256":"b"*64,"structural_report_sha256":"c"*64}
    args.train_admission.write_text(json.dumps(admission))
    config={"schema":bench.TRAINING_SCHEMA,"dataset":"Sand","objective":"faithful","arm":"mix","seed":0,
        "updates":100000,"batch_size":2,"history":6,"noise_std":6.7e-4,"checkpoint_every":10000,"log_every":1,
        "initialization":"from scratch; paired seed across arms; empty Adam; no parent checkpoint",
        "research_protocol_sha256":bench.sha(args.training_protocol),"graph_exposure":trainer.graph_exposure_config("mix"),
        "source_sha256":{**trainer.SOURCE_PINS,"train_sand_graph_support_cuda.py":bench.TRAINER_SHA},
        "data":{"admission_sha256":bench.sha(args.train_admission),"manifest_sha256":admission["manifest_sha256"],
            "metadata_sha256":bench.METADATA_SHA,"frames_per_trajectory":320,"particle_type_ids":[6],
            "converter_sha256":admission["converter_sha256"],"structural_report_sha256":admission["structural_report_sha256"],
            "source":{"sha256":bench.SOURCES["train"][2]}},
        "architecture":{"width":128,"message_passing_blocks":10,"mlp_layers":2},
        "graph":{"radius":.015,"backend":"scipy_host","cap":128,"self_candidates":True,"augmentation_probability":0.},
        "optimizer":{"name":"Adam","initial_lr":1e-4,"final_lr":1e-5,"decay_updates":100000,"betas":[.9,.999],
            "eps":1e-8,"weight_decay":0.,"foreach":False,"fused":False,"gradient_clipping":None},
        "runtime":{"deterministic_algorithms":True,"deterministic_warn_only":False,"cublas_workspace_config":":4096:8",
            "tf32":False,"amp":False,"compile":False,"ddp":False}}
    history={"training":[],"elapsed_seconds":1.,"graph_updates":[]}
    for step in range(512):
        examples=[{"example_slot":slot,"exposure_coin":False,"expanded":False,"annulus_pairs":4,
            "optional_budget_if_exposed":1,"selected_optional_pairs":0,"native_edge_sha256":"1"*64,
            "optional_pair_sha256":"2"*64,"noisy_current_sha256":"3"*64,
            "coin_seed_material":[20261005,0,step,slot,4409],"pair_seed_material":[20261005,0,step,slot,5501]} for slot in (0,1)]
        history["graph_updates"].append({"completed_steps":step+1,"absolute_schedule_step":step,
            "frame_ids":["synthetic:0:6","synthetic:1:7"],"noise_sha256":"4"*64,"examples":examples})
    payload={"format_version":2,"cuda_sand_graph_support_schema":bench.TRAINING_SCHEMA,"run_config":config,
        "run_config_sha256":bench.canonical_hash(config),"completed_steps":512,"history":history,
        "training_config":{"loss":"faithful","cuda_sand_graph_support_run":config,"completed_optimizer_updates":512}}
    return args,admission,payload


def test_default_description_has_no_scientific_imports():
    path=ROOT/"benchmark_sand_graph_support_rollout.py"
    program="""import builtins,runpy,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','numpy','scipy','gns','research'):raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    output=subprocess.run([sys.executable,"-I","-c",program,str(path)],text=True,capture_output=True,check=True)
    assert json.loads(output.stdout)["execution"] is False


def test_cli_only_admits_explicit_512_timing_and_no_test_or_initialized_model():
    args=["--execute","--repo","r","--manifest","m","--admission","a","--structural-report","s","--protocol","p",
        "--output-dir","o","--split","valid","--model-kind","checkpoint","--arm","base","--seed","0",
        "--train-admission","ta","--training-protocol","tp","--timing-release","tr","--checkpoint","c",
        "--checkpoint-sha256","1"*64,"--checkpoint-updates","512","--trainer-source","ts"]
    assert bench.parse_args(args).checkpoint_updates == 512
    for extra in (["--checkpoint-updates","100000"],["--split","test"],["--model-kind","initialized"],
                  ["--objective","nll"],["--mode","locked-test"],["--seed","3"]):
        with pytest.raises(SystemExit):bench.parse_args(args+extra)


def test_checkpoint_accepts_own_512_schema_and_rejects_old_or_different_identity(tmp_path):
    args,admission,payload=checkpoint_fixture(tmp_path)
    assert bench.check_checkpoint_contract(payload,args,admission,bench.TRAINER_SHA)["arm"] == "mix"
    for mutation in (lambda p:p.update(cuda_sand_training_schema="old"),
                     lambda p:p.update(full_training_schema="old"),lambda p:p.update(graph_support_schema=1),
                     lambda p:p.update(completed_steps=100000),lambda p:p["history"]["graph_updates"].pop(),
                     lambda p:p["history"]["graph_updates"][0].update(absolute_schedule_step=9)):
        value=copy.deepcopy(payload);mutation(value)
        with pytest.raises(ValueError):bench.check_checkpoint_contract(value,args,admission,bench.TRAINER_SHA)
    changed=SimpleNamespace(**{**vars(args),"arm":"base"})
    with pytest.raises(ValueError):bench.check_checkpoint_contract(payload,changed,admission,bench.TRAINER_SHA)


def test_checkpoint_rejects_recipe_protocol_exposure_and_admission_changes(tmp_path):
    args,admission,payload=checkpoint_fixture(tmp_path)
    for mutate in (lambda c:c.update(log_every=100),lambda c:c.update(research_protocol_sha256="0"*64),
                   lambda c:c["graph_exposure"].update(optional_pair_fraction=.5),
                   lambda c:c["data"].update(admission_sha256="0"*64),
                   lambda c:c["source_sha256"].update({"research/faithful_graph_support.py":"0"*64}),
                   lambda c:c["optimizer"].update(foreach=True)):
        value=copy.deepcopy(payload);mutate(value["run_config"])
        value["run_config_sha256"]=bench.canonical_hash(value["run_config"])
        with pytest.raises(ValueError):bench.check_checkpoint_contract(value,args,admission,bench.TRAINER_SHA)


def test_timing_release_binds_scope_and_all_sources(tmp_path):
    args=SimpleNamespace(arm="base",seed=0,checkpoint_sha256="1"*64,diagnostic_mode="full-rollout")
    for name in ("timing_release","trainer_source","protocol","training_protocol","train_admission"):
        setattr(args,name,tmp_path/(name+".txt"));getattr(args,name).write_text(name)
    release={"schema":"adaptgns_sand_graph_support_timing_release_v1","status":"admitted_for_timing_only",
        "scientific_training_admission":False,"arm":"base","seed":0,"checkpoint_updates":512,
        "checkpoint_sha256":args.checkpoint_sha256,"policies":list(bench.POLICIES),"diagnostic_mode":"full-rollout"}
    for key,path in (("trainer_sha256",args.trainer_source),("evaluation_source_sha256",ROOT/"benchmark_sand_graph_support_rollout.py"),
                     ("protocol_sha256",args.protocol),("training_protocol_sha256",args.training_protocol),
                     ("training_admission_sha256",args.train_admission)):release[key]=bench.sha(path)
    args.timing_release.write_text(json.dumps(release));bench.check_timing_release(args)
    for key,value in (("scientific_training_admission",True),("checkpoint_updates",100000),("checkpoint_sha256","0"*64),
                      ("policies",list(bench.POLICIES[:-1])),("evaluation_source_sha256","0"*64)):
        args.timing_release.write_text(json.dumps({**release,key:value}))
        with pytest.raises(ValueError):bench.check_timing_release(args)


def test_all_six_full_rollout_dispatch_and_failed_policy_null_mean(tmp_path,monkeypatch):
    monkeypatch.setattr(bench,"DEADLINE",datetime.now(timezone.utc)+timedelta(days=1))
    calls=[]
    def rollout(model,positions,types,metadata,method,horizon,seed,device,trace_steps):
        calls.append((method,horizon,seed))
        complete=method != "dense"
        row={"policy":method,"horizon":horizon,"status":"complete" if complete else "failed",
            "failure":None if complete else {"category":"candidate_pair_resource_guard"},
            "completed_steps":horizon if complete else 1,"mse_per_step":[1.]*horizon if complete else [1.],
            "mean_rollout_mse":1. if complete else None}
        return row,{"trace":np.zeros((1,2),dtype=np.float32)}
    native=SimpleNamespace(np=np,full=SimpleNamespace(synchronize=lambda device:None),rollout=rollout)
    args=SimpleNamespace(output_dir=tmp_path,mode="full-rollout",result_schema="final_synthetic",max_seconds=100,seed=2,arm="base",objective="faithful")
    manifest={"metadata":{},"records":[{"positions":{"shape":[9,3,2]}}]}
    cases=[{"source_index":0,"trajectory_id":"synthetic:0","size_group":"complete_final_split"}]
    rows,state=bench.run_cases(native,None,[(np.zeros((9,3,2)),np.full(3,6))],manifest,cases,args,"cpu","a"*64,bench.time.perf_counter())
    assert len(rows) == len(calls) == 6 and state == "complete_with_guard_failures"
    assert [call[0] for call in calls] == list(bench.POLICIES)
    result=json.loads((tmp_path/"result.json").read_text())
    assert result["schema"] == "final_synthetic" and "All30" in result["interpretation"]
    assert result["summary"]["dense"]["equal_case_mean_rollout_mse"] is None


@pytest.mark.parametrize("mode",["same-state","clean-validation"])
def test_diagnostic_timing_dispatches_fixed_observed_frames_without_a_final_cohort(tmp_path,monkeypatch,mode):
    monkeypatch.setattr(bench,"DEADLINE",datetime.now(timezone.utc)+timedelta(days=1))
    calls=[]
    def one(*args):
        item=args[5]
        calls.append(item.copy())
        return {**item,"status":"complete","failure":None},{"synthetic":np.zeros(2,dtype=np.float32)}
    class Loader:
        def create_module(self,spec):return None
        def exec_module(self,module):
            module.same_state=one;module.clean_validation=one
            module.summarize=lambda expected,rows,kind:{"expected_frames":len(expected),"returned_frames":len(rows)}
    original_spec=bench.importlib.util.spec_from_file_location
    monkeypatch.setattr(bench.importlib.util,"spec_from_file_location",
        lambda name,path:bench.importlib.util.spec_from_loader(name,Loader()) if name=="_sand_graph_support_timing_diagnostics" else original_spec(name,path))
    args=SimpleNamespace(output_dir=tmp_path,diagnostic_mode=mode,split="valid",checkpoint_updates=512,max_seconds=100,seed=0,arm="mix")
    helpers=SimpleNamespace(np=np,torch=SimpleNamespace(no_grad=contextlib.nullcontext))
    manifest={"metadata":{},"records":[{"positions":{"shape":[320,2,2]}} for _ in range(3)]}
    cases=[{"source_index":i,"trajectory_id":f"synthetic:{i}","size_group":size} for i,size in enumerate(("small","median","large"))]
    sync=[]
    native=SimpleNamespace(full=SimpleNamespace(synchronize=lambda device:sync.append(device)))
    rows,state=bench.run_timing_diagnostics(native,helpers,None,[(None,None)]*3,manifest,cases,args,"cpu","a"*64,bench.time.perf_counter())
    assert state=="complete" and len(rows)==len(calls)==15
    assert [item["target_frame"] for item in calls[:5]]==[7,85,163,241,319]
    assert len(sync)==30
    assert all(row["synchronized_diagnostic_call_seconds_including_parity_warmup_audits"]>=0
               and row["numeric_artifact_publication_seconds"]>=0 for row in rows)
    result=json.loads((tmp_path/"result.json").read_text())
    assert result["scope"]=="non-test512 infrastructure timing only" and result["timing_forecast_admitted"] is False
