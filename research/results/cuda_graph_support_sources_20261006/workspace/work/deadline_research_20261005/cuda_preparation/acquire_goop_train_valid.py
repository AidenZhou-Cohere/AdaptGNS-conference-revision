#!/usr/bin/env python3
"""Acquire only the three pinned public Goop train/valid objects; preserve failures."""
import argparse,base64,datetime,hashlib,json,os,time
from pathlib import Path
BASE='https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/Goop/'
SOURCES={'metadata.json':(362,'1599153763201175','9pwq+g=='),'train.tfrecord':(3358535518,'1599154403717337','OrpMIQ=='),'valid.tfrecord':(93682221,'1599153784596039','EqHuFw==')}
META_SHA='565d6e13be91be6a6b0fbc31a9aa19ed70a5505228411d0928288fcf874ca3dd'
def write(path,value):
 temporary=path.with_suffix(path.suffix+'.tmp')
 with temporary.open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 temporary.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output-dir',type=Path);a=p.parse_args()
 if not a.execute: print(json.dumps({'execute':False,'dataset':'Goop','sources':SOURCES,'test_access':False}));return
 if a.output_dir is None:p.error('--output-dir is required')
 import requests,crc32c
 if crc32c.crc32c(b'123456789')!=0xe3069283:raise ValueError('CRC32C implementation self-test failed')
 a.output_dir.mkdir(parents=True,exist_ok=False)
 started=time.perf_counter();report={'schema':'official_goop_train_valid_acquisition_v1','status':'running','dataset':'Goop','source_family':'official_gns_tfrecord','files':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scientific_training_admission':False}
 current=None
 try:
  for name,(size,generation,expected_crc) in SOURCES.items():
   url=BASE+name;current={'name':name,'saved_name':name,'url':url,'status':'started','received_bytes':0};report['files'].append(current);write(a.output_dir/'acquisition_report.json',report)
   with requests.get(url,params={'generation':generation},headers={'Accept-Encoding':'identity'},stream=True,timeout=(20,120),allow_redirects=False) as response:
    current['response_status']=response.status_code;current['response_headers']={k:v for k,v in response.headers.items() if k.lower() in {'content-length','content-type','content-encoding','x-goog-generation','x-goog-hash','etag','last-modified'}}
    if response.status_code!=200:raise ValueError('Expected200 without redirect: '+name)
    if response.headers.get('x-goog-generation')!=generation or int(response.headers.get('Content-Length','-1'))!=size or response.headers.get('Content-Encoding','identity')!='identity':raise ValueError('Generation, size or encoding mismatch: '+name)
    server_hashes={part.strip().split('=',1)[0]:part.strip().split('=',1)[1] for part in response.headers.get('x-goog-hash','').split(',') if '=' in part}
    if server_hashes.get('crc32c')!=expected_crc:raise ValueError('PublisherCRC mismatch: '+name)
    digest=hashlib.sha256();crc=0;count=0;partial=a.output_dir/(name+'.partial')
    with partial.open('xb') as f:
     for block in response.iter_content(chunk_size=1<<20):
      if not block:continue
      f.write(block);digest.update(block);crc=crc32c.crc32c(block,crc);count+=len(block);current['received_bytes']=count
      if count>size:raise ValueError('Received more than pinned bytes: '+name)
     f.flush();os.fsync(f.fileno())
    encoded=base64.b64encode(crc.to_bytes(4,'big')).decode();current.update(generation=generation,sha256=digest.hexdigest(),crc32c_base64=encoded,crc32c_verified=encoded==expected_crc)
    if count!=size or encoded!=expected_crc or (name=='metadata.json' and digest.hexdigest()!=META_SHA):raise ValueError('Complete source checksum mismatch: '+name)
    final=a.output_dir/name
    if final.exists():raise FileExistsError(final)
    partial.rename(final);current['status']='complete';write(a.output_dir/'acquisition_report.json',report)
  report['status']='complete'
 except BaseException as error:
  report.update(status='failed',error_type=type(error).__name__,error=str(error))
  if current is not None and current['status']!='complete':current['status']='failed'
  raise
 finally:
  report['elapsed_seconds']=time.perf_counter()-started;report['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(a.output_dir/'acquisition_report.json',report)
if __name__=='__main__':main()
