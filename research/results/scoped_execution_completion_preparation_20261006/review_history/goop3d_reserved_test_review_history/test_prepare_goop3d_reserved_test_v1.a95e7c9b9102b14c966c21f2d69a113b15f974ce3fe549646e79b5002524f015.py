"""Synthetic TFRecords, scalar cohort and fake HTTP only. No official test or GPU access."""
import base64
import contextlib
from datetime import datetime,timedelta,timezone
import hashlib
import io
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import pytest
import prepare_goop3d_reserved_test_v1 as M

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
READER=ROOT/'outputs/AdaptGNS/research/prepare_full_waterdrop.py'
METADATA=HERE.parent/'extension_feasibility_20261006/Goop-3D_metadata_json.json'
CONTEXT=HERE/'goop3d_context_semantics_review.json'
reader=M.load(READER,M.READER_SHA,'_synthetic3d_test_reader');np=reader.np


def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2)+'\n');return path


def args(**overrides):
    names=('cohort','cohort_audit','root_release','source_metadata','metadata','output','output_dir','input_dir',
        'acquisition_report','reader','numeric_root','train_manifest','valid_manifest','admission','context_semantics',
        'auxiliary_report','cross_split_audit','protocol','trainer_source','data_loader')
    return SimpleNamespace(**{**dict.fromkeys(names),**overrides})


def authorize(a):
    r={'schema':M.GATE_SCHEMA,'status':'approved_for_'+a.mode,'mode':a.mode,'issued_by':'root','issued_utc':M.now(),
        'cohort_sha256':M.sha(a.cohort),'cohort_audit_sha256':M.sha(a.cohort_audit),'endpoint_updates':37,
        'preparation_source_sha256':M.sha(M.__file__),'source_url':M.URL,'all_six_checkpoint_hashes_verified':True,
        'test_converter_independently_reviewed':True,'whole_invocation_outer_timeout_required':True,
        'review_rationale':'Explicitly synthetic test fixture only; never actual admission',
        'output_path':str((a.output if a.mode in ('inspect','preflight') else a.output_dir).resolve()),
        'files_sha256':{str(p):M.sha(p) for p in M.input_paths(a).values()}}
    if a.source_metadata:r['source_metadata_sha256']=M.sha(a.source_metadata)
    put(a.root_release,r);M.gate(a);return r


@pytest.fixture
def gate_args(tmp_path):
    models=[{'arm':a,'seed':s,'objective':'faithful','completed_steps':37,'graph_history_updates':37,
        'checkpoint_sha256':str(1+s+(3 if a=='mix' else 0))*64,'all_optimizer_steps_equal_endpoint':True,
        'all_state_and_moments_finite':True,'source_data_protocol_verified':True,'checkpoint_bytes_verified':True}
        for a in ('base','mix') for s in range(3)]
    common={'training_schema':'adaptgns_goop3d_graph_support_cuda_training_v2','endpoint_updates':37,
        'protocol_sha256':M.PROTOCOL_SHA,'trainer_sha256':M.TRAINER_SHA,
        'graph_sha256':'ca5898ec3f487d8138adf7479743156874a7796df09841f8ff7711b97536ad50',
        'models':models,'issued_by':'root','evaluation_admitted':True}
    audit={**common,'schema':'adaptgns_goop3d_graph_support_complete_cohort_audit_v2','status':'all_six_endpoints_and_pairing_verified',
        'paired_seeds':[{'seed':s,**{k:True for k in ('initial_model_tensor_identity','initial_cpu_cuda_rng_identity','initial_empty_adam_identity',
        'all_frame_noise_lr_schedules_equal','all_graph_budgets_and_rng_material_verified')}} for s in range(3)]}
    ap=put(tmp_path/'audit.json',audit)
    cp=put(tmp_path/'cohort.json',{**common,'schema':'adaptgns_goop3d_graph_support_final_cohort_v2','status':'frozen_for_final_evaluation',
        'cohort_audit_sha256':M.sha(ap),'created_utc':(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()})
    a=args(mode='inspect',cohort=cp,cohort_audit=ap,root_release=tmp_path/'release.json',output=tmp_path/'headers.json',
        protocol=HERE/'goop3d_scientific_protocol_v1.md',trainer_source=HERE/'train_goop3d_graph_support_cuda_v2.py')
    authorize(a);return a


def test_default_no_access(monkeypatch):
    monkeypatch.setattr(M,'gate',lambda *a:pytest.fail('must not gate/read'))
    out=io.StringIO()
    with contextlib.redirect_stdout(out):assert M.main([])==0
    assert json.loads(out.getvalue())['test_accessed'] is False


def test_prospective_endpoint_gate_is_not_fixed100k(gate_args):
    c,r=M.gate(gate_args);assert c['endpoint_updates']==r['endpoint_updates']==37


@pytest.mark.parametrize('which',['partial','pair','schema','mode','source','review','future','closure'])
def test_gate_refuses_before_test_request(gate_args,monkeypatch,which):
    a=gate_args
    if which in ('partial','schema','future'):
        c=M.read(a.cohort)
        if which=='partial':c['models'][0]['completed_steps']=36
        elif which=='schema':c['schema']='adaptgns_goop3d_vectorized_capacity_checkpoint_v1'
        else:c['created_utc']=(datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat()
        put(a.cohort,c)
    elif which=='pair':
        x=M.read(a.cohort_audit);x['paired_seeds'][0]['initial_empty_adam_identity']=False;put(a.cohort_audit,x)
        c=M.read(a.cohort);c['cohort_audit_sha256']=M.sha(a.cohort_audit);put(a.cohort,c)
    r=M.read(a.root_release)
    r.update(cohort_sha256=M.sha(a.cohort),cohort_audit_sha256=M.sha(a.cohort_audit),files_sha256={str(p):M.sha(p) for p in M.input_paths(a).values()})
    if which=='mode':r['mode']='acquire'
    elif which=='source':r['source_url']=M.BASE+'valid.tfrecord'
    elif which=='review':r['test_converter_independently_reviewed']=False
    elif which=='closure':r['files_sha256'].pop(str(a.protocol.resolve()))
    put(a.root_release,r)
    monkeypatch.setattr(M,'inspect_source',lambda *a:pytest.fail('must not request'))
    argv=['--execute','--mode','inspect']
    for name in ('cohort','cohort_audit','root_release','protocol','trainer_source','output'):argv+=['--'+name.replace('_','-'),str(getattr(a,name))]
    with pytest.raises(ValueError):M.main(argv)
    assert not a.output.exists()


class Response:
    def __init__(self,raw,fail=False):
        self.raw,self.fail,self.status_code=raw,fail,200
        self.headers={'Content-Length':str(len(raw)),'x-goog-generation':'123456','x-goog-hash':'crc32c='+base64.b64encode(reader.crc32c.crc32c(raw).to_bytes(4,'big')).decode()}
    def __enter__(self):return self
    def __exit__(self,*a):return False
    def iter_content(self,chunk_size):
        yield self.raw[:len(self.raw)//2]
        if self.fail:raise OSError('synthetic stream interruption')
        yield self.raw[len(self.raw)//2:]


class Requests:
    def __init__(self,response):self.response,self.calls=response,[]
    def head(self,url,**kwargs):self.calls.append(('HEAD',url,kwargs));return self.response
    def get(self,url,**kwargs):self.calls.append(('GET',url,kwargs));return self.response


def acquire_args(a,tmp_path,response):
    req=Requests(response);M.inspect_source(a,M.read(a.root_release),req)
    assert req.calls[0][0]=='HEAD' and req.calls[0][2]['allow_redirects'] is False
    a.mode='acquire';a.source_metadata=a.output;a.output_dir=tmp_path/'acquired';a.metadata=METADATA
    r=authorize(a);meta=M.source_metadata(a,r);return a,r,meta


def test_exact_header_and_generation_acquisition(gate_args,tmp_path):
    response=Response(b'fake-source'*100);a,r,meta=acquire_args(gate_args,tmp_path,response);req=Requests(response)
    M.acquire(a,r,meta,req,reader.crc32c)
    out=M.read(a.output_dir/'acquisition_report.json')
    assert out['status']=='complete' and out['files'][1]['sha256']==hashlib.sha256(response.raw).hexdigest()
    assert req.calls[0][2]['params']=={'generation':meta['generation']}
    assert M.read(a.source_metadata)['payload_downloaded'] is False


@pytest.mark.parametrize('bad',['interrupt','generation','crc','oversize','release_changed'])
def test_acquisition_preserves_failure(gate_args,tmp_path,bad):
    response=Response(b'fake-source'*100);a,r,meta=acquire_args(gate_args,tmp_path,response)
    if bad=='interrupt':response.fail=True
    elif bad=='generation':response.headers['x-goog-generation']='99999'
    elif bad=='crc':response.raw=b'wrong'+response.raw[5:]
    elif bad=='oversize':response.raw+=b'x'
    else:put(a.root_release,{**r,'review_rationale':'changed'})
    req=Requests(response)
    with pytest.raises((OSError,ValueError)):M.acquire(a,r,meta,req,reader.crc32c)
    out=M.read(a.output_dir/'acquisition_report.json');assert out['status']=='failed'
    partial=a.output_dir/'test.tfrecord.partial'
    if partial.exists():assert out['files'][1]['retained_partial_sha256']==M.sha(partial)
    assert len(req.calls)==(0 if bad=='release_changed' else 1)


def payload(index,frames=301,aux=True):
    ex=reader.example_pb2.SequenceExample();ex.context.feature['key'].int64_list.value.extend([91+index,7])
    ex.context.feature['particle_type'].bytes_list.value.append(np.array([7,7],dtype='<i8').tobytes())
    positions=(np.arange(frames*6,dtype='<f4').reshape(frames,2,3)/100000+.25+index/100).astype('<f4')
    for p in positions:ex.feature_lists.feature_list['position'].feature.add().bytes_list.value.append(p.tobytes())
    if aux:
        bits=np.array([0x80000000,0x7fc12345],dtype='<u4')
        for _ in range(frames):ex.feature_lists.feature_list['step_context'].feature.add().bytes_list.value.append(bits.tobytes())
    return ex.SerializeToString()


def encoded(values):
    raw=b''
    for p in values:
        size=struct.pack('<Q',len(p));raw+=size+struct.pack('<I',reader.masked_crc(size))+p+struct.pack('<I',reader.masked_crc(p))
    return raw


def conversion(a,tmp_path,values):
    response=Response(encoded(values));a,r,meta=acquire_args(a,tmp_path,response);M.acquire(a,r,meta,Requests(response),reader.crc32c)
    a.mode='convert';a.input_dir=a.output_dir;a.acquisition_report=a.input_dir/'acquisition_report.json'
    a.output_dir=tmp_path/'numeric';a.reader=READER;r=authorize(a)
    return a,r,meta


@pytest.fixture
def converted(gate_args,tmp_path):
    a,r,meta=conversion(gate_args,tmp_path,[payload(i,aux=(i!=1)) for i in range(3)])
    M.convert(a,r,meta);return a,r,meta


def test_actual_count_and_source_key_auxiliary_bits_preserved(converted):
    a,r,meta=converted;m=M.read(a.output_dir/'test.json');s=M.read(a.output_dir/'structural_report.json')
    assert m['record_count']==3 and s['status']=='complete_structural_only' and m['records'][0]['source_key']==[91,7]
    assert 'step_context' not in m['records'][1]
    aux=np.load(a.output_dir/m['records'][0]['step_context']['path'],allow_pickle=False)
    assert aux.view('<u4')[0].tolist()==[0x80000000,0x7fc12345]
    assert s['converter_helper_sha256']==M.CONVERTER_SHA


@pytest.mark.parametrize('bad',['shape','duplicate','crc'])
def test_conversion_adverse_structure_retained(gate_args,tmp_path,bad):
    values=[payload(0,frames=300)] if bad=='shape' else [payload(0),payload(0)] if bad=='duplicate' else [payload(0)]
    a,r,meta=conversion(gate_args,tmp_path,values)
    if bad=='crc':
        p=a.input_dir/'test.tfrecord';raw=bytearray(p.read_bytes());raw[-1]^=1;p.write_bytes(raw)
    with pytest.raises(Exception):M.convert(a,r,meta)
    assert M.read(a.output_dir/'structural_report.json')['status']=='failed'
    assert not (a.output_dir/'test.json').exists()
    assert (a.input_dir/'test.tfrecord').exists()
    if bad=='shape':assert M.read(a.output_dir/'.test.staging/preserved_records.json')[0]['positions']['shape'][0]==300


def test_numeric_content_identity_is_independent_of_id(converted):
    a,_,_=converted;records=M.read(a.output_dir/'test.json')['records']
    assert M.numeric_identity(a.output_dir,records[0],np)!=M.numeric_identity(a.output_dir,records[1],np)
    assert M.numeric_identity(a.output_dir,records[0],np)==M.numeric_identity(a.output_dir,{**records[0],'id':'other'},np)


def prior_split(root,split,count,shift):
    records=[];(root/'metadata.json').write_bytes(METADATA.read_bytes())
    for i in range(count):
        record={'id':f'{split}:{i:06d}','source_index':i}
        for key,v in {'positions':np.full((301,1,3),shift+i/100000,dtype='<f4'),'particle_types':np.array([7],dtype='<i8')}.items():
            p=root/split/f'{key}_{i:06d}.npy';p.parent.mkdir(exist_ok=True);np.save(p,v,allow_pickle=False)
            record[key]={'path':str(p.relative_to(root)),'sha256':M.sha(p),'size_bytes':p.stat().st_size,'shape':list(v.shape),'dtype':v.dtype.str}
        record['trajectory_content_sha256']=hashlib.sha256((record['positions']['sha256']+':'+record['particle_types']['sha256']).encode()).hexdigest();records.append(record)
    return put(root/(split+'.json'),{'dataset':'Goop-3D','split':split,'metadata':M.read(METADATA),'metadata_sha256':M.METADATA_SHA,'record_count':count,'records':records})


def census_args(converted,tmp_path,monkeypatch):
    a,r,meta=converted;prior=tmp_path/'prior';prior.mkdir();a.train_manifest=prior_split(prior,'train',1000,.31);a.valid_manifest=prior_split(prior,'valid',100,.41)
    monkeypatch.setattr(M,'TRAIN_MANIFEST_SHA',M.sha(a.train_manifest));monkeypatch.setattr(M,'VALID_MANIFEST_SHA',M.sha(a.valid_manifest))
    a.mode='census';a.numeric_root=a.output_dir;a.output_dir=tmp_path/'census';a.context_semantics=CONTEXT;r=authorize(a)
    return a,r,meta


def test_all_split_census_candidate_then_frozen_preflight(converted,tmp_path,monkeypatch):
    a,r,meta=census_args(converted,tmp_path,monkeypatch);M.census(a,r,meta)
    aux=M.read(a.output_dir/'auxiliary_report.json');cross=M.read(a.output_dir/'cross_split_audit.candidate.json');candidate=M.read(a.output_dir/'final_test_admission.candidate.json')
    assert cross['record_counts']=={'train':1000,'valid':100,'test':3} and cross['duplicate_pairs']==[]
    assert aux['splits']['test']['records'][0]['unique_float32_bits_hex']==['7fc12345','80000000']
    assert aux['splits']['test']['auxiliary_absent_count']==1 and candidate['issued_by']=='preparation_wrapper_not_root'
    # Synthetic root promotion only, distinct files; keep all candidate bytes.
    a.cross_split_audit=put(tmp_path/'root_cross.json',{**cross,'status':'all_required_splits_verified','issued_by':'root'})
    a.admission=put(tmp_path/'root_test_admission.json',{**candidate,'status':'admitted','issued_by':'root','cross_split_audit_sha256':M.sha(a.cross_split_audit)})
    a.auxiliary_report=a.output_dir/'auxiliary_report.json';a.mode='preflight';a.output=tmp_path/'preflight.json'
    a.data_loader=ROOT/'outputs/AdaptGNS/adaptive-gns/gns/data_loader.py'
    original=M.load
    def synthetic_pin_load(path,pin,name):
        module=original(path,pin,name)
        if Path(path).name=='evaluate_goop3d_graph_support_v1.py':
            module.TRAIN_MANIFEST_SHA=M.sha(a.train_manifest);module.VALID_MANIFEST_SHA=M.sha(a.valid_manifest)
        return module
    monkeypatch.setattr(M,'load',synthetic_pin_load);r=authorize(a);M.preflight(a,r,meta)
    report=M.read(a.output);assert report['status']=='frozen_split_and_evidence_contract_passed'
    assert len(report['source_order_grid'])==3 and report['same_state_scheduled_histories']==15 and not report['test_evaluation_executed']


def test_cross_split_duplicate_is_retained_before_refusal(converted,tmp_path,monkeypatch):
    a,r,meta=census_args(converted,tmp_path,monkeypatch)
    train=M.read(a.train_manifest);valid=M.read(a.valid_manifest)
    for key in ('positions','particle_types'):valid['records'][0][key]=dict(train['records'][0][key])
    valid['records'][0]['trajectory_content_sha256']=train['records'][0]['trajectory_content_sha256'];put(a.valid_manifest,valid)
    monkeypatch.setattr(M,'VALID_MANIFEST_SHA',M.sha(a.valid_manifest));r=authorize(a)
    with pytest.raises(ValueError,match='duplicates retained'):M.census(a,r,meta)
    cross=M.read(a.output_dir/'cross_split_audit.candidate.json')
    assert cross['status']=='failed_duplicates_retained' and len(cross['duplicate_pairs'])==1
    assert not (a.output_dir/'final_test_admission.candidate.json').exists()


@pytest.mark.parametrize('header,value',[('Content-Length','0'),('x-goog-generation','not-generation'),('x-goog-hash','md5=abc'),('Content-Encoding','gzip')])
def test_publisher_header_failure_retains_observed_headers(gate_args,header,value):
    response=Response(b'fake');response.headers[header]=value;req=Requests(response)
    with pytest.raises(ValueError):M.inspect_source(gate_args,M.read(gate_args.root_release),req)
    assert M.read(gate_args.output)['response_headers'][header]==value
    assert M.read(gate_args.output)['status']=='failed' and len(req.calls)==1


def test_tfrecord_payload_crc_failure_retained_despite_good_object_crc(gate_args,tmp_path):
    raw=bytearray(encoded([payload(0)]));raw[-1]^=1;response=Response(bytes(raw))
    a,r,meta=acquire_args(gate_args,tmp_path,response);M.acquire(a,r,meta,Requests(response),reader.crc32c)
    a.mode='convert';a.input_dir=a.output_dir;a.acquisition_report=a.input_dir/'acquisition_report.json'
    a.output_dir=tmp_path/'numeric';a.reader=READER;r=authorize(a)
    with pytest.raises(ValueError,match='TFRecord payload CRC mismatch'):M.convert(a,r,meta)
    assert M.read(a.output_dir/'structural_report.json')['status']=='failed'
    assert 'CRC' in M.read(a.output_dir/'.test.staging/failure.json')['reason']
    assert M.read(a.acquisition_report)['status']=='complete'
