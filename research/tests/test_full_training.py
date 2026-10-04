from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.full_training import (assert_finite_gradients, evaluate_validation,
    host_noise, learning_rate, restore_checkpoint, sample_indices,
    save_checkpoint, validation_schedule, RunLock, recover_initialization,
    tensors_are_finite, config_hash)


def test_step_schedule_is_independent_of_process_rng_and_resume_position():
    uninterrupted = [sample_indices(2,step,995000) for step in range(12)]
    np.random.seed(915)
    np.random.normal(size=200)
    torch.manual_seed(915)
    torch.randn(200)
    resumed = [sample_indices(2,step,995000) for step in range(7,12)]
    assert resumed == uninterrupted[7:]
    assert all(len(indices)==2 and min(indices)>=0 and max(indices)<995000 for indices in uninterrupted)
    assert uninterrupted != [sample_indices(3,step,995000) for step in range(12)]


def test_host_noise_is_paired_and_preserves_global_rng_and_kinematic_particles():
    state = torch.get_rng_state().clone()
    kinds = torch.tensor([0,3,0])
    first = host_noise((3,6,2),kinds,1,12)
    torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
    torch.randn(200)
    second = host_noise((3,6,2),kinds,1,12)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert torch.count_nonzero(first[:,0]) == 0
    assert torch.count_nonzero(first[1]) == 0
    assert torch.count_nonzero(first[0]) > 0
    assert not torch.equal(first,host_noise((3,6,2),kinds,1,13))


def test_validation_schedule_represents_all_30_trajectories_with_128_fixed_frames():
    lengths = list(range(995,1025))
    dataset = SimpleNamespace(_data_lengths=lengths)
    ids = [f"valid:{i:06d}" for i in range(30)]
    schedule = validation_schedule(dataset,ids)
    assert len(schedule)==128
    assert len({row['id'] for row in schedule})==128
    assert {row['trajectory'] for row in schedule}==set(ids)
    assert sorted(sum(row['trajectory']==name for row in schedule) for name in ids)==[4]*22+[5]*8
    assert schedule==validation_schedule(dataset,ids)
    cumulative=np.cumsum(lengths)
    for row in schedule:
        trajectory=ids.index(row['trajectory'])
        offset=0 if trajectory==0 else cumulative[trajectory-1]
        assert row['dataset_index']==offset+row['target_frame']-6


def test_learning_rate_uses_declared_100k_clock_even_for_a_smoke_prefix():
    assert learning_rate(0)==1e-4
    assert learning_rate(99999)==pytest.approx(1e-5)
    assert learning_rate(100001)==pytest.approx(1e-5)
    assert 9.99e-5<learning_rate(3)<1e-4


def tiny_model():
    return torch.nn.Sequential(torch.nn.Linear(2,7),torch.nn.Tanh(),torch.nn.Linear(7,2))


def tiny_updates(model,optimizer,start,stop):
    x=torch.arange(40,dtype=torch.float32).reshape(20,2)/40
    for step in range(start,stop):
        indices=sample_indices(5,step,len(x))
        features=x[indices]
        for group in optimizer.param_groups:group['lr']=learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        loss=(model(features)-features.square()).square().mean()
        loss.backward()
        assert_finite_gradients(model)
        optimizer.step()


def test_atomic_checkpoint_restores_exact_adam_continuation_and_rejects_changed_config(tmp_path):
    torch.manual_seed(213)
    full=tiny_model()
    interrupted=deepcopy(full)
    full_optim=torch.optim.Adam(full.parameters(),lr=1e-4)
    interrupted_optim=torch.optim.Adam(interrupted.parameters(),lr=1e-4)
    config={'objective':'nll','steps':8,'data_sha256':'verified-data','source_sha256':'verified-source'}
    tiny_updates(full,full_optim,0,8)
    tiny_updates(interrupted,interrupted_optim,0,3)
    path=tmp_path/'checkpoint.pt'
    pointer=save_checkpoint(path,interrupted,interrupted_optim,config,3,{'training':[]},'cpu')
    assert pointer['completed_steps']==3
    assert not path.with_suffix('.pt.tmp').exists()
    resumed=tiny_model()
    resumed_optim=torch.optim.Adam(resumed.parameters(),lr=.5)
    step,history=restore_checkpoint(path,resumed,resumed_optim,config,'cpu')
    assert step==3 and history=={'training':[]}
    tiny_updates(resumed,resumed_optim,step,8)
    for actual,expected in zip(resumed.parameters(),full.parameters()):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for actual,expected in zip(resumed_optim.state.values(),full_optim.state.values()):
        for key in expected:
            torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    with pytest.raises(ValueError,match='hashes differ'):
        restore_checkpoint(path,resumed,resumed_optim,{**config,'data_sha256':'changed'},'cpu')


def test_nonfinite_gradient_is_rejected_before_optimizer_update():
    model=torch.nn.Linear(2,1)
    before=deepcopy(model.state_dict())
    for param in model.parameters():param.grad=torch.full_like(param,float('nan'))
    with pytest.raises(FloatingPointError,match='Nonfinite'):
        assert_finite_gradients(model)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)


def test_clean_validation_uses_equal_trajectory_weights_and_preserves_model_mode():
    class Dummy(torch.nn.Module):
        def forward(self,**kwargs):
            assert torch.count_nonzero(kwargs['position_sequence_noise'])==0
            target=kwargs['next_positions']
            return torch.zeros_like(target),torch.ones(len(target)),target
    class Dataset:
        def __getitem__(self,index):
            target=np.full((1,2),1. if index<2 else 2.,dtype=np.float32)
            return ((np.zeros((1,6,2),dtype=np.float32),np.array([0]),1),target)
    schedule=[{'dataset_index':0,'trajectory':'a','id':'a:6','target_frame':6},
              {'dataset_index':1,'trajectory':'a','id':'a:7','target_frame':7},
              {'dataset_index':2,'trajectory':'b','id':'b:6','target_frame':6}]
    model=Dummy().train()
    result=evaluate_validation(model,Dataset(),schedule,torch.device('cpu'))
    assert model.training
    assert result['equal_trajectory_mean']['coordinate_mse']==pytest.approx(2.5)
    assert result['binned_vector_se_calibration_gap']==pytest.approx(3.)
    assert len(result['calibration_bins'])==1
    assert result['calibration_bins'][0]['weight']==pytest.approx(1.)


def test_exclusive_run_lock_rejects_live_owner_and_releases_after_error(tmp_path):
    with pytest.raises(ValueError,match='intentional'):
        with RunLock(tmp_path) as process:
            assert process['pid']>0 and process['started_utc']
            with pytest.raises(RuntimeError,match='already locked'):
                with RunLock(tmp_path):
                    pass
            with pytest.raises(RuntimeError,match='live process'):
                with RunLock(tmp_path,clear_stale=True):
                    pass
            assert (tmp_path/'run.lock').exists()
            raise ValueError('intentional')
    assert not (tmp_path/'run.lock').exists()
    assert not (tmp_path/'run.lock.recovery').exists()


def test_stale_lock_removal_requires_explicit_flag_and_dead_pid(tmp_path,monkeypatch):
    import json
    import research.full_training as module
    (tmp_path/'run.lock').write_text(json.dumps({'pid':12345,'token':'old'}))
    with pytest.raises(RuntimeError,match='already locked'):
        with RunLock(tmp_path):
            pass
    checked=[]
    def dead(pid):
        checked.append(pid)
        return False
    monkeypatch.setattr(module,'pid_alive',dead)
    with RunLock(tmp_path,clear_stale=True):
        assert checked==[12345]
    assert not (tmp_path/'run.lock').exists()


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),float('-inf')])
def test_flat_finite_guard_checks_every_parameter_and_preserves_values(bad):
    arrays=[torch.ones(2,3),torch.ones(1),torch.ones(4)]
    assert tensors_are_finite(arrays)
    arrays[-1][-1]=bad
    assert not tensors_are_finite(arrays)
    assert not tensors_are_finite([])


def test_only_unstarted_runs_can_recover_missing_checkpoint_pointer(tmp_path):
    import json
    config={'objective':'nll','steps':8}
    assert recover_initialization(tmp_path,config) is None
    (tmp_path/'status.json').write_text(json.dumps({'completed_steps':0,'run_config_sha256':config_hash(config)}))
    model=tiny_model();optimizer=torch.optim.Adam(model.parameters())
    save_checkpoint(tmp_path/'checkpoint-000000.pt',model,optimizer,config,0,{'training':[]},'cpu')
    pointer=recover_initialization(tmp_path,config)
    assert pointer['completed_steps']==0 and (tmp_path/'latest.json').exists()
    (tmp_path/'latest.json').unlink()
    (tmp_path/'status.json').write_text(json.dumps({'completed_steps':1,'run_config_sha256':config_hash(config)}))
    with pytest.raises(ValueError,match='evidence of training'):
        recover_initialization(tmp_path,config)
