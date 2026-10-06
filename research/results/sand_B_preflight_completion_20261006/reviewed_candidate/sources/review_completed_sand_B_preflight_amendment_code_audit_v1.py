import pathlib,json,hashlib,datetime,sys
P=pathlib.Path(__file__).resolve().parent;CA=P/'sand_reserved_released_1808_root_v1';role='B';op=sys.argv[1];assert op in ['preflight_valid','preflight_test'];prefix=op;C=P/'sand_B_preflight_amendment_released_root_v1';rd=lambda p:json.loads(pathlib.Path(p).read_bytes());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();utc=datetime.datetime.fromisoformat;R='/root/repos/AdaptGNS-cuda-20261006/';checks=[]
def ck(n,b):
 if not b:raise AssertionError(n)
 checks.append(n)
r=rd(C/(prefix+'.cpu_release.json'));m=rd(C/(prefix+'.mode_release.json'));phase=rd(C/'amendment_phase.json');cap=rd(C/(prefix+'.collection.json'));ext=rd(C/(prefix+'.external.json'));own=r['owner_output_dir'];out=r['paths']['--output-dir'];rows={x['path']:x for x in cap['files']};paths={p:C/'collected'/p.removeprefix(R) for p in rows};t=rd(paths[own+'/owner_terminal.json']);started=rd(paths[own+'/owner_started.json']);stop=utc(phase['amendment_stop_utc']);now=datetime.datetime.now(datetime.timezone.utc)
original=rd(C/(prefix+'.root_original_external_exit.json'))
ck('Root originaltool zero observed before collection',original['recorded_by']=='root' and original['original_tool_exit_code']==0 and type(original['original_tool_session']) is int and utc(ext['observed_utc'])<=utc(original['observed_utc'])<=utc(cap['checked_utc'])<stop)
ck('Explicit new600 phase and originalsource unrestarted',phase['new_shared_amendment_seconds']==600 and (stop-utc(phase['amendment_started_utc'])).total_seconds()==600 and phase['original_source_clock_restarted'] is False and phase['new_or_restarted_clock_granted'] is True and r['amendment_phase']['sha256']==sha(C/'amendment_phase.json'))
ck('Actual releasecore uses same amended alarm',m['amendment_phase_sha256']==sha(C/'amendment_phase.json') and m['preparation_started_utc']==phase['amendment_started_utc'] and m['preparation_stop_utc']==phase['amendment_stop_utc'])
ck('Original external process exited0 reaped no timeout',ext['exit_code']==0 and ext['local_transport_reaped'] is True and ext['local_transport_timeout'] is False and ext['signals_to_own_local_group']==[])
ck('Original successful worker complete and nativeabsent',t['status']=='complete' and t['failure'] is None and t['child']['exit_code']==0 and t['child']['reaped'] is True and t['child_native_absent'] is True and t['child']['signals']==[] and t['child']['cleanup_errors']==[])
ck('Actual originalrelease and exactcommand',t['release_sha256']==sha(C/(prefix+'.cpu_release.json')) and t['child']['command']==r['command'])
ck('Complete exact externalnative registry',cap['native_absent']=={str(p):True for p in [t['outer_timeout_identity']['pid'],t['owner_identity']['pid'],t['child']['pid']]} and len(cap['native_absent'])==3)
ck('Correct host noGPU',cap['hostname']==r['hostname'] and cap['gpu_apps']=='')
ck('Amendment phase ordering and collection before its original shared stop',utc(phase['amendment_started_utc'])<=utc(ext['started_utc'])<=utc(t['publication_utc'])<=utc(ext['observed_utc'])<=utc(cap['checked_utc'])<stop and now<stop)
ck('Root capture maps equal terminal',cap['input_sha256']==t['input_sha256'] and cap['output_sha256']==t['output_sha256'])
for p,h in r['inputs_sha256'].items():ck('All actual releaseinput retained '+p,t['input_sha256'].get(p)==h)
ck('Same sourcephase originalhash',t['input_sha256'][r['source_phase']['path']]==sha(CA/'source_phase.json'))
for p,row in rows.items():
 if pathlib.PurePosixPath(p).suffix not in ['.npy','.npz']:ck('Captured raw scalar bytes '+p,paths[p].is_file() and not paths[p].is_symlink() and sha(paths[p])==row['sha256'] and paths[p].stat().st_size==row['bytes'])
 else:ck('Opaque numeric evidence only '+p,not paths[p].exists() and row['bytes']>0 and len(row['sha256'])==64)
for k in ['output_sha256','evidence_sha256']:
 for p,h in t[k].items():ck('Completedowner '+k+' '+p,p in rows and rows[p]['sha256']==h)
ck('Exact5 owner files',len([p for p in rows if p.startswith(own+'/')])==5)
mode=m['mode'];names={'acquire':['metadata.json','test.npz','download_report.json','root_preparation_receipt.json'],'convert':['metadata.json','structural_report.json','test.json']+[f'test/{s}_{i:06d}.npy' for i in range(30) for s in ['position','type']],'census':['all_split_census.json'],'candidate':['split_admission_candidate.json'],'preflight':['split_preflight.json']}[mode]
ck('Exactsuccess output tree',set(t['output_sha256'])=={out+'/'+n for n in names} and len(rows)==5+len(names))
ck('Native sourceguard original hardstop retained',t['native_hard_guard']==started['native_hard_guard'] and t['native_hard_guard']['absolute_deadline_ns']==r['hard_deadline_monotonic_ns'] and t['native_hard_guard']['signal']=='SIGKILL' and t['native_hard_guard']['deleted_before_exit'] is False)
time=t['native_outer_timing'];hard=r['hard_deadline_monotonic_ns'];computed=time['computed_monotonic_ns'];ck('Dynamicnative outer duration and5secondbootstrap',time['term_seconds']==(hard-computed)//10**9-20 and 0<=t['native_hard_guard']['armed_monotonic_ns']-computed<=5*10**9 and time['bootstrap_allowance_seconds']==5)
ck('Publication inside originalnative bounds',t['publication_monotonic']*1e9<time['earliest_relative_kill_monotonic_ns']<hard)
ck('No owner scientificadmission claim',t['scientific_admission'] is False and t['root_original_tool_exit_and_timeout_owner_child_native_closure_required'] is True)
ck('Numeric input count correct for mode',t['numeric_input_files_bound']==(2120 if mode in ['census','candidate','preflight'] else 0))
if mode=='acquire':
 rep=rd(paths[out+'/download_report.json']);rec=rd(paths[out+'/root_preparation_receipt.json']);fr={x['name']:x for x in rep['files']}
 ck('Acquisition originalfixed schema/status/dataset',rep['schema']=='sand_public_acquisition_v2' and rep['status']=='complete_with_object_dtypes' and rep['dataset']=='Sand')
 ck('Exactly2requested originalfiles',rep['requested_files']==['metadata.json','test.npz'] and set(fr)=={'metadata.json','test.npz'} and len(rep['files'])==2)
 ck('Original fixeddownloader andmetadata',rep['downloader_sha256']=='864d9c974deabceb19b5f3b8e88ced1ea7281595a0d3a526e8d39f382aa4605a' and rep['metadata_reference_sha256']=='cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0')
 for name,size in [('metadata.json',363),('test.npz',85825802)]:
  f=fr[name];ck('Originalacquired '+name+' bytes/hash/status',f['status']=='downloaded_and_inspected' and f['stage']=='complete' and f['saved_name']==name and f['received_bytes']==f['expected_bytes']==rows[out+'/'+name]['bytes']==size and f['received_sha256']==rows[out+'/'+name]['sha256'])
 ck('All30ZIPheaders inspected; fullCRC remains conversion duty',fr['test.npz']['zip']['member_count']==30 and fr['test.npz']['zip']['full_member_crc_verified'] is False)
 ck('No decoded/conversion/metrics orpublisherhash overclaim',all(rep[k] is False for k in ['publisher_npz_sha256_available','arrays_decoded','pickle_used','conversion_performed','metrics_computed','numeric_payload_validity_verified']))
 ck('Rootreceipt exact cohort/core/report lineage',rec['cohort_sha256']==phase['cohort_sha256'] and rec['root_release_sha256']==sha(C/(prefix+'.mode_release.json')) and rec['download_report_sha256']==sha(paths[out+'/download_report.json']) and rec['arrays_decoded'] is False and rec['test_evaluation_executed'] is False and rec['publisher_test_sha256_available'] is False)
elif mode=='convert':
 manifest=rd(paths[out+'/test.json']);structural=rd(paths[out+'/structural_report.json']);details=structural['splits']['test'];acquire=rd(CA/'acquire.collection.json');acqout=R+'sand_reserved_preparation_20261006_v1/acquired/'
 ck('Complete convertedtest scalar contract',manifest['format']=='gns-trajectory-manifest' and manifest['dataset']=='Sand' and manifest['split']=='test' and manifest['record_count']==len(manifest['records'])==30)
 ck('Exact convertedmetadata bytes',manifest['metadata_sha256']==rows[out+'/metadata.json']['sha256']=='cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0')
 ck('Originalacquired source identity andCRC result',manifest['source']['sha256']==acquire['output_sha256'][acqout+'test.npz'] and manifest['source']['size_bytes']==85825802 and manifest['source']['member_count']==30 and manifest['source']['ZIP_CRC_verified'] is True and manifest['source']['acquisition_report_sha256']==acquire['output_sha256'][acqout+'download_report.json'])
 ck('Complete structural31source scalars',structural['schema']=='designsafe_sand_numeric_repackage_v1' and structural['status']=='complete_structural_only' and structural['cohort_sha256']==phase['cohort_sha256'] and structural['root_release_sha256']==sha(C/(prefix+'.mode_release.json')) and set(structural['splits'])=={'test'})
 ck('Complete T320float32type6 scalar summary',details['record_count']==len(details['records'])==30 and details['frame_lengths']==[320] and details['position_dtypes']==['<f4'] and details['particle_type_ids']==[6] and details['kinematic_type3_particles']==0 and details['manifest_sha256']==sha(paths[out+'/test.json']))
 ck('Crosssplit duplicate census explicitly remains required',structural['duplicate_trajectories_within_and_across_selected_splits'] is False and 'full cross-split census still required' in structural['duplicate_scope'])
 logical=[]
 for i,record in enumerate(manifest['records']):
  ck('Convertedrecord exactindex '+str(i),record['source_index']==i and record['id']==f'test:{i:06d}')
  logical.append(record['logical_content_sha256'])
  for key,kind,dtype in [('positions','position','<f4'),('particle_types','type','<i8')]:
   desc=record[key];q=out+'/'+desc['path'];ck('Convertedopaque descriptor '+str(i)+key,desc['path']==f'test/{kind}_{i:06d}.npy' and desc['dtype']==dtype and desc['sha256']==rows[q]['sha256'] and desc['size_bytes']==rows[q]['bytes'])
  shape=record['positions']['shape'];ck('Convertedposition geometry '+str(i),len(shape)==3 and shape[0]==320 and shape[1]>0 and shape[2]==2)
 ck('30 distinct converted logical trajectory identities',len(set(logical))==30)
elif mode=='census':
 c=rd(paths[out+'/all_split_census.json']);counts={'train':1000,'valid':30,'test':30}
 ck('Census complete original1060 trajectories',c['schema']=='adaptgns_sand_complete_numeric_census_v1' and c['status']=='complete_no_duplicates' and c['counts']==counts and len(c['records'])==1060)
 ck('Census exactcohort/core/preparer and noeval',c['cohort_sha256']==phase['cohort_sha256'] and c['root_release_sha256']==sha(C/(prefix+'.mode_release.json')) and c['preparation_source_sha256']=='6a40fe9196157c0666ffe556eb0e92781e9fc3b84f0c45b5ff67e09be0fce1b8' and c['evaluation_executed'] is False and c['all_split_numeric_duplicates_checked'] is True)
 ck('All2120 originalnumeric byte bindings protected',len(c['numeric_files_sha256'])==2120 and all(t['input_sha256'].get(p)==h for p,h in c['numeric_files_sha256'].items()))
 ck('All1060 logical trajectory hashes unique',len({x['logical_content_sha256'] for x in c['records']})==1060)
 for s,count in counts.items():
  ck(s+' exactcensus manifest binding',c['manifests_sha256'][s]==r['inputs_sha256'][r['paths']['--'+s+'-manifest']])
  ck(s+' complete originalsource index denominator',[x['source_index'] for x in c['records'] if x['split']==s]==list(range(count)))
elif mode in ['candidate','preflight']:
 split=op.split('_')[1];q=rd(paths[out+'/'+('split_admission_candidate.json' if mode=='candidate' else 'split_preflight.json')]);census_path=r['paths']['--census-report'];cp=CA/'collected'/census_path.removeprefix(R);census=rd(cp)
 ck('Original completecensus retained',r['inputs_sha256'][census_path]==sha(cp) and q['census_sha256']==sha(cp) and len(census['numeric_files_sha256'])==2120 and all(t['input_sha256'].get(p)==h for p,h in census['numeric_files_sha256'].items()))
 ck('Exactsplit manifeststructure lineage',q['split']==split and q['manifest_sha256']==r['inputs_sha256'][r['paths']['--'+split+'-manifest']] and q['structural_report_sha256']==r['inputs_sha256'][r['paths']['--structural-report']])
 ck('Exact frozenpreparer source lineage',q['preparation_source_sha256']=='6a40fe9196157c0666ffe556eb0e92781e9fc3b84f0c45b5ff67e09be0fce1b8')
 if mode=='candidate':
  ck('Candidate remains rootunadmitted',q['schema']=='adaptgns_sand_graph_support_final_evaluation_admission_v1' and q['status']=='candidate_requires_root_review' and q['issued_by']=='preparation_wrapper_not_root' and q['root_preparation_release_sha256']==sha(C/(prefix+'.mode_release.json')))
  ck('Candidate exact T320type6float32count30',q['dataset']=='Sand' and q['cohort_manifest_sha256']==phase['cohort_sha256'] and q['frames_per_trajectory']==320 and q['record_count']==30 and q['particle_type_ids']==[6] and q['position_dtype']=='<f4' and q['metadata_sha256']=='cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0')
 else:
  ck('Preflight actualcomplete scopedstatus noeval',q['schema']=='adaptgns_sand_final_split_preflight_v1' and q['status']=='complete_split_contract_passed' and q['evaluation_executed'] is False and q['all_split_numeric_duplicates_checked'] is True)
  ck('Preflight exactcohort/core/admission bindings',q['cohort_sha256']==phase['cohort_sha256'] and q['root_release_sha256']==sha(C/(prefix+'.mode_release.json')) and q['admission_sha256']==r['inputs_sha256'][r['paths']['--admission']])
  ck('Preflight exact30frames320horizon314evaluator',q['record_count']==30 and q['frames']==320 and q['horizon']==314 and q['evaluator_sha256']=='952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58')

for p,row in rows.items():
 if paths[p].exists():ck('Final retainedscalar '+p,sha(paths[p])==row['sha256'])
result={'schema':'sand_completed_B_preflight_amendment_independent_review_v1','reviewer':'code_audit','checked_utc':now.isoformat(),'status':'passed_original_mode_outputs_nativeclosure_and_scalar_semantics','host_role':role,'operation':op,'check_count':len(checks),'checks':checks,'operational_release_sha256':sha(C/(prefix+'.cpu_release.json')),'mode_release_sha256':sha(C/(prefix+'.mode_release.json')),'original_source_phase_sha256':sha(CA/'source_phase.json'),'amendment_phase_sha256':sha(C/'amendment_phase.json'),'original_tool_exit_receipt_sha256':sha(C/(prefix+'.root_original_external_exit.json')),'outputs_sha256':t['output_sha256'],'native_absent':cap['native_absent'],'amendment_stop_utc':stop.isoformat(),'remaining_seconds':(stop-now).total_seconds(),'evidence_sha256':{str(C/(prefix+s)):sha(C/(prefix+s)) for s in ['.collection.json','.external.json']},'scope':'Read completed original scalar captures only; numeric files retained as original remote byte/hash evidence. No array deserialization, replay, training/evaluation or remoteoperation performed.','limitations':['Acquisition publisher NPZ SHA unavailable: acquisition report preserves observed exactbytes/SHA andZIPheaders; full member CRC remains conversion duty and publisher checksum authentication is unavailable.','Native bootstrap assumes previously stated ordinary five-second scheduling allowance; new shared600-second amendment stop and both preflight/collection/review times remain shared; original900-second history remains incomplete.']}
f=P/('sand_completed_B_amendment_'+prefix+'_independent_review_code_audit_v1.json')
with f.open('x') as w:json.dump(result,w,indent=2,sort_keys=True);w.write('\n')
print(json.dumps({'path':str(f),'sha256':sha(f),'checks':len(checks),'remaining_seconds':result['remaining_seconds'],'status':result['status']}))
