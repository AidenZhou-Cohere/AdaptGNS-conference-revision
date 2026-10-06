"""Offline source/control and tiny numeric fixtures; no official arrays or network."""
import copy
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import repackage_designsafe_sand as C
from test_repackage_designsafe_sand import envelope,member,make_zip,META
from test_supervise_sand_final_evaluation_scoped_v1 import fixture as eval_fixture,split_fixture,put,h

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sand_reserved_scoped',HERE/'prepare_sand_reserved_test_scoped_v1.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
NOW=datetime(2026,10,6,23,0,tzinfo=timezone.utc)


def gate_fixture(tmp_path,monkeypatch,mode='acquire'):
    _,_,_=eval_fixture(tmp_path,monkeypatch)
    cp=tmp_path/'cohort.json';ap=tmp_path/'audit.json'
    cohort,audit=M.read(cp),M.read(ap)
    common={'issued_by':'root','cohort_id':'synthetic_scoped_sand','adapter_sha256':M.COHORT_ADAPTER_SHA}
    audit.update(common);put(ap,audit)
    cohort.update(common,created_utc=(NOW-timedelta(hours=1)).isoformat(),cohort_audit_sha256=M.sha(ap),
        operational_amendment_sha256=M.AMENDMENT_SHA,schedule_plan_sha256=M.PLAN_SHA)
    put(cp,cohort)
    args=SimpleNamespace(mode=mode,cohort=cp,cohort_audit=ap,root_release=tmp_path/'root_release.json',output_dir=tmp_path/'preparation')
    names=('prepare_sand_reserved_test_scoped_v1.py','prepare_sand_final_cohort_scoped_v1.py','download_sand_public_v2.py',
        'repackage_designsafe_sand.py','evaluate_sand_graph_support_final.py','benchmark_sand_graph_support_rollout.py',
        'sand_graph_support_100k_protocol_v1.md','sand_scoped_operational_amendment_v1.md','sand_scoped_schedule_fixed_spec_v2.json')
    release={'schema':M.RELEASE_SCHEMA,'issued_by':'root','status':'approved_for_'+mode,'preparation_source_sha256':M.sha(M.__file__),
        'cohort_sha256':M.sha(cp),'cohort_audit_sha256':M.sha(ap),'output_dir':str(args.output_dir),'no_retry_or_checkpoint_selection':True,
        'preparation_started_utc':NOW.isoformat(),'preparation_stop_utc':(NOW+timedelta(seconds=900)).isoformat(),
        'issued_utc':NOW.isoformat(),'clock_error_bound_seconds':1,'files_sha256':{str(p):M.sha(p) for p in [cp,ap]+[HERE/n for n in names]}}
    put(args.root_release,release);monkeypatch.setattr(M,'now',lambda:NOW)
    return args,release


def manual_args(tmp_path,monkeypatch,**kw):
    monkeypatch.setattr(M,'now',lambda:NOW)
    return SimpleNamespace(output_dir=tmp_path/'output',_bindings={},_stop=NOW+timedelta(seconds=900),
        _cohort_sha=h('synthetic-cohort'),_release_sha=h('synthetic-release'),**kw)


def bind(a,*paths):a._bindings.update({str(p):M.sha(p) for p in paths})


def test_default_does_not_load_or_acquire(monkeypatch,capsys):
    monkeypatch.setattr(M,'gate',lambda a:pytest.fail('default must not access cohort'))
    monkeypatch.setattr(M,'load',lambda *a:pytest.fail('default must not load'))
    assert M.main([])==0
    assert json.loads(capsys.readouterr().out)['all_modes_share_preparation_window_seconds']==900


def test_complete_cohort_and_source_gate(tmp_path,monkeypatch):
    a,r=gate_fixture(tmp_path,monkeypatch)
    M.gate(a);assert a._cohort_sha==r['cohort_sha256'] and a._stop==NOW+timedelta(seconds=899)


def test_incomplete_cohort_fails_before_test_or_network(tmp_path,monkeypatch):
    a,r=gate_fixture(tmp_path,monkeypatch);cohort=M.read(a.cohort);cohort['models'][0]['completed_steps']=99999;put(a.cohort,cohort)
    r['files_sha256'][str(tmp_path/'UNACQUIRED_RESERVED_TEST')]='a'*64;put(a.root_release,r)
    monkeypatch.setattr(M,'sha',lambda p:pytest.fail('incomplete cohort must fail before input/source access'))
    with pytest.raises(ValueError,match='six distinct'):M.gate(a)


@pytest.mark.parametrize('bad',['long_window','late_window','wrong_mode','checkpoint_duplicate','wrong_mapping_adapter'])
def test_changed_admission_refused(tmp_path,monkeypatch,bad):
    a,r=gate_fixture(tmp_path,monkeypatch)
    if bad=='long_window':r['preparation_stop_utc']=(NOW+timedelta(seconds=901)).isoformat()
    elif bad=='late_window':
        r['preparation_started_utc']='2026-10-06T23:30:00+00:00';r['preparation_stop_utc']='2026-10-06T23:45:00+00:00'
    elif bad=='wrong_mode':r['status']='approved_for_convert'
    else:
        c=M.read(a.cohort)
        if bad=='checkpoint_duplicate':c['models'][0]['checkpoint_sha256']=c['models'][1]['checkpoint_sha256']
        else:c['adapter_sha256']='0'*64
        put(a.cohort,c)
    put(a.root_release,r)
    with pytest.raises(ValueError):M.gate(a)


def test_exact_test_only_downloader_dispatch(tmp_path,monkeypatch):
    a=manual_args(tmp_path,monkeypatch);calls=[]
    def fake_main(argv):
        calls.append(argv);a.output_dir.mkdir();put(a.output_dir/'download_report.json',{'status':'complete_with_object_dtypes','files':[]});return 2
    monkeypatch.setattr(M,'load',lambda *args:SimpleNamespace(main=fake_main))
    monkeypatch.setattr(M,'checked_acquisition',lambda *args:{})
    M.acquisition(a)
    assert calls==[['--output',str(a.output_dir),'--files','test.npz','--max-seconds','900']]
    assert M.read(a.output_dir/'root_preparation_receipt.json')['arrays_decoded'] is False


def synthetic_archive(root,count=30,duplicate=False):
    root.mkdir();values=[]
    for i in range(count):
        p=np.full((320,3,2),.2+(.0001*i if not duplicate else 0),dtype=np.float32)
        values.append(envelope(p,np.full(3,6,dtype=np.int64)))
    row=make_zip(root/'test.npz',[(f'simulation_trajectory_{i}.npy',member(v)) for i,v in enumerate(values)])
    put(root/'metadata.json',META);report=put(root/'download_report.json',{'synthetic':True})
    return row,report,values


@pytest.mark.parametrize('duplicate',[False,True])
def test_reuses_restricted_converter_and_preserves_failed_staging(tmp_path,monkeypatch,duplicate):
    source=tmp_path/'source';row,report,values=synthetic_archive(source,duplicate=duplicate)
    a=manual_args(tmp_path,monkeypatch,input_dir=source,acquisition_report=report);bind(a,report,source/'metadata.json',source/'test.npz')
    monkeypatch.setattr(M,'checked_acquisition',lambda *args:row)
    if duplicate:
        with pytest.raises(ValueError,match='Duplicate'):M.convert(a)
        assert (a.output_dir/'.test.staging/failure.json').exists() and not (a.output_dir/'test.json').exists()
    else:
        M.convert(a);manifest=M.read(a.output_dir/'test.json');assert len(manifest['records'])==30
        for i in (0,29):
            saved=np.load(a.output_dir/manifest['records'][i]['positions']['path'],allow_pickle=False)
            assert saved.tobytes()==values[i][0].tobytes()
        assert M.read(a.output_dir/'structural_report.json')['converter_sha256']==M.CONVERTER_SHA
    assert M.sha(source/'test.npz')==row['received_sha256']


def numeric_manifest(root,split,count,start):
    target=root/split;target.mkdir(parents=True);rows=[]
    for i in range(count):
        p=np.full((320,1,2),float(start+i),dtype=np.float32);t=np.full(1,6,dtype=np.int64)
        pd=C.save_array(np,target/f'p{i}.npy',split,p);td=C.save_array(np,target/f't{i}.npy',split,t)
        rows.append({'id':f'{split}:{i:06d}','source_index':i,'positions':pd,'particle_types':td,
            'logical_content_sha256':h(C.value_hash(p)+':'+C.value_hash(t))})
    return put(root/(split+'.json'),{'dataset':'Sand','split':split,'record_count':count,'metadata_sha256':M.METADATA_SHA,'records':rows})


@pytest.mark.parametrize('duplicate',[False,True])
def test_complete_numeric_census_and_cross_split_duplicate(tmp_path,monkeypatch,duplicate):
    paths={s:numeric_manifest(tmp_path,s,n,offset) for s,n,offset in [('train',1000,0),('valid',30,1000),('test',30,0 if duplicate else 1030)]}
    a=manual_args(tmp_path,monkeypatch,**{s+'_manifest':p for s,p in paths.items()});bind(a,*paths.values())
    monkeypatch.setattr(M,'TRAIN_MANIFEST_SHA',M.sha(paths['train']))
    if duplicate:
        with pytest.raises(ValueError,match='Duplicate numeric'):M.census(a)
        assert not a.output_dir.exists()
    else:
        M.census(a);result=M.read(a.output_dir/'all_split_census.json')
        assert result['counts']=={'train':1000,'valid':30,'test':30} and len(result['records'])==1060
        assert len(result['numeric_files_sha256'])==2120


@pytest.mark.parametrize('split',['valid','test'])
def test_candidate_root_sign_preflight_uses_frozen_evaluator(tmp_path,monkeypatch,split):
    a=manual_args(tmp_path,monkeypatch,split=split,protocol=HERE/'sand_graph_support_100k_protocol_v1.md',train_admission=HERE/'sand_train_admission.json')
    sources={s:split_fixture(s,tmp_path/'splits',a._cohort_sha) for s in ('valid','test')}
    train=put(tmp_path/'splits/train.json',{'dataset':'Sand','split':'train','metadata_sha256':M.METADATA_SHA,'record_count':1000,
        'records':[{'id':f'train:{i:06d}','source_index':i,**{k:{'path':f'train/{k}{i}.npy'} for k in ('positions','particle_types')}} for i in range(1000)]})
    paths={'train':train,**{s:v[0] for s,v in sources.items()}};files={}
    for name,path in paths.items():
        manifest=M.read(path)
        for i,record in enumerate(manifest['records']):
            for key in ('positions','particle_types'):
                numeric=path.parent/record[key]['path'];numeric.parent.mkdir(exist_ok=True)
                numeric.write_bytes(f'opaque {name} {key} {i}'.encode());record[key]['sha256']=M.sha(numeric);files[str(numeric)]=M.sha(numeric)
            if name!='train':record['trajectory_content_sha256']=h(record['positions']['sha256']+':'+record['particle_types']['sha256'])
        put(path,manifest)
        if name!='train':
            structural=M.read(sources[name][1]);structural['splits'][name]['manifest_sha256']=M.sha(path);put(sources[name][1],structural)
    a.train_manifest=train;a.valid_manifest=paths['valid'];a.test_manifest=paths['test'];a.structural_report=sources[split][1]
    monkeypatch.setattr(M,'TRAIN_MANIFEST_SHA',M.sha(train));numeric=Path(next(iter(files)))
    a.census_report=put(tmp_path/'census.json',{'schema':'adaptgns_sand_complete_numeric_census_v1','status':'complete_no_duplicates',
        'cohort_sha256':a._cohort_sha,'counts':M.COUNTS,'all_split_numeric_duplicates_checked':True,
        'preparation_source_sha256':M.sha(M.__file__),'manifests_sha256':{s:M.sha(p) for s,p in paths.items()},'numeric_files_sha256':files})
    bind(a,*paths.values(),a.structural_report,a.census_report,a.protocol,a.train_admission)
    M.candidate(a);candidate=a.output_dir/'split_admission_candidate.json';a.admission=candidate;bind(a,candidate);a.output_dir=tmp_path/'preflight'
    with pytest.raises(ValueError,match='Separate root admission'):M.preflight(a)
    signed=M.read(candidate);signed.update(issued_by='root',status='admitted_for_final_evaluation')
    a.admission=put(tmp_path/'root_split_admission.json',signed);bind(a,a.admission)
    M.preflight(a);result=M.read(a.output_dir/'split_preflight.json')
    assert result['evaluator_sha256']==M.EVALUATOR_SHA and result['horizon']==314 and result['evaluation_executed'] is False
    original=M.read(a.census_report)
    for field in ('numeric_files_sha256','manifests_sha256'):
        partial=copy.deepcopy(original);partial[field].pop(next(iter(partial[field])));put(a.census_report,partial);bind(a,a.census_report)
        a.output_dir=tmp_path/('partial_'+field)
        with pytest.raises(ValueError,match='census'):M.candidate(a)
        assert not a.output_dir.exists()
    put(a.census_report,original);bind(a,a.census_report)
    numeric.write_bytes(b'changed after completed census');a.output_dir=tmp_path/'changed_numeric_preflight'
    with pytest.raises(ValueError,match='Numeric/output binding'):M.preflight(a)
    assert not (a.output_dir/'split_preflight.json').exists()


def test_bound_json_is_parsed_from_hashed_bytes(tmp_path,monkeypatch):
    a=manual_args(tmp_path,monkeypatch);p=put(tmp_path/'bound.json',{'value':1});bind(a,p)
    monkeypatch.setattr(M,'sha',lambda p:pytest.fail('no separate hash/read race'))
    assert M.bound(a,p)=={'value':1}
    p.write_text('{"value":2}')
    with pytest.raises(ValueError,match='root-bound'):M.bound(a,p)


def test_shared_deadline_and_input_mutation_refuse_publication(tmp_path,monkeypatch):
    a=manual_args(tmp_path,monkeypatch);p=put(tmp_path/'source.json',{'v':1});bind(a,p)
    monkeypatch.setattr(M,'now',lambda:a._stop)
    with pytest.raises(ValueError,match='window'):M.stable(a)
    monkeypatch.setattr(M,'now',lambda:NOW);put(p,{'v':2})
    with pytest.raises(ValueError,match='binding'):M.stable(a)


@pytest.mark.parametrize('bad',[None,'hash','members','extra'])
def test_complete_acquisition_scalar_contract(tmp_path,monkeypatch,bad):
    # Only sizes/hash constants for these tiny opaque bytes are substituted.
    (tmp_path/'metadata.json').write_bytes(b' '*363);(tmp_path/'test.npz').write_bytes(b'opaque synthetic ZIP placeholder')
    monkeypatch.setattr(M,'METADATA_SHA',M.sha(tmp_path/'metadata.json'))
    monkeypatch.setattr(M,'TEST_BYTES',(tmp_path/'test.npz').stat().st_size)
    rows=[]
    for name in ('metadata.json','test.npz'):
        p=tmp_path/name;rows.append({'name':name,'saved_name':name,'status':'downloaded_and_inspected','stage':'complete',
            'received_bytes':p.stat().st_size,'expected_bytes':p.stat().st_size,'received_sha256':M.sha(p),'zip':{'member_count':30}})
    report={'schema':'sand_public_acquisition_v2','status':'complete_with_object_dtypes','dataset':'Sand','downloader_sha256':M.DOWNLOADER_SHA,
        'metadata_reference_sha256':M.METADATA_SHA,'requested_files':['metadata.json','test.npz'],'files':rows}
    if bad=='hash':rows[1]['received_sha256']='0'*64
    elif bad=='members':rows[1]['zip']['member_count']=29
    elif bad=='extra':report['requested_files'].append('train.npz')
    if bad:
        with pytest.raises(ValueError):M.checked_acquisition(tmp_path,report)
    else:assert M.checked_acquisition(tmp_path,report)==rows[1]


@pytest.mark.parametrize('bad',['input','numeric','clock'])
def test_ready_publication_checks_after_serialization(tmp_path,monkeypatch,bad):
    a=manual_args(tmp_path,monkeypatch);source=put(tmp_path/'source.json',{'value':1});bind(a,source)
    numeric=tmp_path/'numeric.npy';numeric.write_bytes(b'synthetic');outputs={str(numeric):M.sha(numeric)}
    original=json.dumps
    def encode(value,**kw):
        raw=original(value,**kw)
        if bad=='input':source.write_bytes(b'changed')
        elif bad=='numeric':numeric.write_bytes(b'changed')
        else:monkeypatch.setattr(M,'now',lambda:a._stop)
        return raw
    monkeypatch.setattr(M.json,'dumps',encode)
    with pytest.raises(ValueError):M.publish(a,tmp_path/'ready.json',{'status':'ready'},outputs)
    assert not (tmp_path/'ready.json').exists()


def test_numeric_identity_rejects_empty_particles(tmp_path):
    root=tmp_path/'test';root.mkdir();p=np.empty((320,0,2),dtype=np.float32);t=np.empty((0,),dtype=np.int64)
    record={'positions':C.save_array(np,root/'p.npy','test',p),'particle_types':C.save_array(np,root/'t.npy','test',t),
        'logical_content_sha256':h(C.value_hash(p)+':'+C.value_hash(t))}
    with pytest.raises(ValueError,match='T320/D2/type6'):M.numeric_identity(np,C,tmp_path,record)
