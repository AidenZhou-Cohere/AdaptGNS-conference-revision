"""Independent inert simulator probes of numerical control flow; no real model/data."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location('independent_methods_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
D=load('run_goop_action_gate_v1');C=load('goop_global_action_gate_core_v1')
assert hashlib.sha256((HERE/'goop_global_action_gate_core_v1.py').read_bytes()).hexdigest()=='d3986c2ac3786e9fc0d36d76339bfeded5448497f7dad9b0559efaaaa53a43ca'


class Guard(Exception):
    def __init__(self):self.details={'category':'state_divergence'}


class Simulator:
    def __init__(self,fail_at=None):
        self.calls=[];self.rng_draws=[];self.parity_calls=0;self.fail_at=fail_at
        self.full=SimpleNamespace(RolloutGuard=Guard,check_state=self.check,synchronize=lambda device:None,
            state_hash=lambda x:hashlib.sha256(np.asarray(x).tobytes()).hexdigest(),boundary_metrics=lambda x,b:{'synthetic_max':float(np.max(x))})
        self.bridge=SimpleNamespace(MAX_ABS=10,supplied=self.supplied,validate_output=self.validate,native_parity=self.parity)
    def check(self,history,*args):
        assert np.isfinite(history).all() and not np.any(history==42),'Target contaminated current history'
    def native_graph(self,history,policy,cached,rng):
        assert cached is None;self.rng_draws.append((policy,float(rng.random())))
        base=np.array([[0,1],[0,1]],dtype=np.int64);optional=np.array([[0,1]],dtype=np.int64) if policy=='random25' else np.empty((0,2),dtype=np.int64)
        edges=np.concatenate((base,np.array([[0,1],[1,0]],dtype=np.int64)),axis=1) if len(optional) else base
        return None,edges,optional,{'retained_optional_pairs':len(optional),'optional_pair_budget':1,'directed_edges':edges.shape[1]}
    def supplied(self,model,history,types,edges,device):
        self.check(history);self.calls.append(history.copy())
        if self.fail_at==len(self.calls):raise Guard()
        increment=np.float32(.001 if edges.shape[1]>2 else .0005)
        return {'prediction':history[-1]+increment,'risk':np.ones(len(types),dtype=np.float32),'raw_risk':np.zeros(len(types),dtype=np.float32)}
    def validate(self,output,n):assert output['prediction'].shape==(n,2)
    def parity(self,model,history,types,edges,device):
        self.parity_calls+=2;self.check(history)
        return {'passed':True},{'synthetic':history.copy()},{'prediction':history[-1].copy(),'risk':np.ones(len(types),dtype=np.float32),'raw_risk':np.zeros(len(types),dtype=np.float32)}


class Positions:
    def __init__(self,simulator,label=False):self.simulator=simulator;self.label=label;self.reads=[]
    def __getitem__(self,index):
        if isinstance(index,slice):return np.zeros((6,2,2),dtype=np.float32)
        expected=2 if self.label else index-5
        assert len(self.simulator.calls)>=expected,'Truth read before required forecast(s)'
        self.reads.append(index);return np.full((2,2),42,dtype=np.float32)


def head(mean=.1):
    return {'schema':C.SCHEMA,'protocol_sha256':C.PROTOCOL_SHA,'feature_names':list(C.FEATURES),'ridge_lambda':.1,
        'training_states':8000,'threshold':0.0,'feature_mean':[0.]*11,'feature_std':[1.]*11,'feature_scale':[1.]*11,
        'coefficients':[0.]*11,'label_mean':mean,'label_std':1.,'label_scale':1.,'validation_requested_expansion_fraction':.5}


def test_label_reads_truth_after_both_actions_and_float64_benefit():
    sim=Simulator();positions=Positions(sim,True)
    row,arrays=D.label_row(sim,C,None,positions,np.array([6,6]),{'source_index':7,'target_frame':62},2,'train','cpu')
    assert row['status']=='complete' and positions.reads==[62] and row['network_passes']==2
    assert np.array_equal(sim.calls[0],sim.calls[1]) and sim.rng_draws[0][1]==sim.rng_draws[1][1]
    expected=np.random.default_rng(np.random.SeedSequence([20261006,27101,2,7,62,1701])).random()
    assert sim.rng_draws[1][1]==expected
    b=float(np.mean((arrays['base_prediction'].astype(np.float64)-42)**2))
    r=float(np.mean((arrays['random25_prediction'].astype(np.float64)-42)**2))
    assert row['base_mse']==b and row['random25_mse']==r and row['signed_benefit']==b-r


@pytest.mark.parametrize('policy',C.POLICIES)
def test_every_policy_runs395_closed_loop_steps_with_frozen_rng_domains(policy):
    sim=Simulator();positions=Positions(sim)
    row,arrays=D.rollout_row(sim,C,None,positions,np.array([6,6]),{'bounds':[[.1,.9],[.1,.9]]},
        {'source_index':4,'policy':policy},1,head(),'test','cpu')
    assert row['status']=='complete' and row['completed_steps']==row['network_passes']==395 and sim.parity_calls==2
    assert positions.reads==list(range(6,401)) and arrays['prediction'].shape==(395,2,2)
    assert np.array_equal(sim.calls[1][-1],arrays['prediction'][0]) and np.array_equal(sim.calls[-1][-1],arrays['prediction'][-2])
    assert row['mse_at_declared_trace_steps']['200']==row['mse_per_step'][199]
    assert row['mean_rollout_mse']==float(np.mean(row['mse_per_step']))
    for step in (1,2,200,395):
        observed=row['steps'][step-1];material=[20261006,27103,1,4,step,1701]
        assert observed['pair_rng_material']==material
        assert sim.rng_draws[step][1]==np.random.default_rng(np.random.SeedSequence(material)).random()
        requested=policy in ('random25','learned_global_gate')
        if policy=='validation_rate_random_gate':
            gate=[20261006,27103,1,4,step,2909]
            requested=np.random.default_rng(np.random.SeedSequence(gate)).random()<.5
            assert observed['gate_rng_material']==gate
        assert observed['requested_expand']==requested


def test_guard_after200_keeps_pointwise200_and_nulls_full395_without_truth_leak():
    sim=Simulator(fail_at=201);positions=Positions(sim)
    row,arrays=D.rollout_row(sim,C,None,positions,np.array([6,6]),{'bounds':[[.1,.9],[.1,.9]]},
        {'source_index':4,'policy':'learned_global_gate'},0,head(),'train_rollout','cpu')
    assert row['status']=='failed' and row['completed_steps']==200 and row['network_passes']==201
    assert row['mean_rollout_mse'] is None and row['mse_at_final_horizon'] is None
    assert row['mse_at_declared_trace_steps']['200']==row['mse_per_step'][199] and positions.reads[-1]==205
    assert row['failure']['forecast_step']==201 and arrays['prediction'].shape[0]==200


def test_nonpositive_head_uses_base_and_gate_never_receives_outcome():
    history=np.zeros((6,2,2),dtype=np.float32)
    decision,features=D.choose_action(C,history,'learned_global_gate',head(0.),'test',0,0,1)
    assert decision['predicted_benefit']==0. and decision['requested_expand'] is False and features.shape==(11,)
    decision,_=D.choose_action(C,history,'learned_global_gate',head(-.1),'test',0,0,1)
    assert decision['requested_expand'] is False
