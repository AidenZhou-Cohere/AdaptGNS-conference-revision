"""Tiny synthetic protobuf/TFRecord tests; no real Goop-3D payload/model access."""
import base64
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import prepare_goop3d_official as prep

ROOT = Path(__file__).resolve().parents[3]
READER_PATH = ROOT / 'outputs/AdaptGNS/research/prepare_full_waterdrop.py'
reader = prep.load_reader(READER_PATH)
np = reader.np


def payload(frames=9, offset=0, auxiliary=False, key=1):
    example = reader.example_pb2.SequenceExample()
    example.context.feature['key'].int64_list.value.extend([key, 100 + key])
    types = np.array([0, 3, 8], dtype='<i8')
    example.context.feature['particle_type'].bytes_list.value.append(types.tobytes())
    positions = (np.arange(frames*9, dtype=np.float32).reshape(frames,3,3) / 1000 + offset)
    positions[0,0,0] = np.float32(-0.0)
    for frame in positions:
        example.feature_lists.feature_list['position'].feature.add().bytes_list.value.append(frame.tobytes())
    if auxiliary:
        bits = np.array([0x7fc12345, 0x80000000], dtype='<u4').view('<f4')
        for _ in range(frames):
            example.feature_lists.feature_list['step_context'].feature.add().bytes_list.value.append(bits.tobytes())
    return example.SerializeToString(), positions, types


def write_records(path, values):
    with path.open('wb') as stream:
        for raw in values:
            size = struct.pack('<Q', len(raw))
            stream.write(size + struct.pack('<I', reader.masked_crc(size)) + raw + struct.pack('<I', reader.masked_crc(raw)))
    raw = path.read_bytes()
    return {'received_bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest(),
            'crc32c_base64':base64.b64encode(reader.crc32c.crc32c(raw).to_bytes(4,'big')).decode()}


class Goop3DPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='goop3d_converter_synthetic_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root/'out'
        self.output.mkdir()
        self.metadata = {'dim':3,'sequence_length':300,'default_connectivity_radius':.025}

    def convert(self, values, split='train', seen=None):
        path = self.root/(split+'.tfrecord')
        row=write_records(path,values)
        records,summary=prep.convert_one(reader,path,split,row,self.output,self.metadata,set() if seen is None else seen)
        return records,summary,path,row

    def test_description_only_has_no_reader_import_or_input_reads(self):
        with mock.patch.object(prep,'load_reader',side_effect=AssertionError('must not import')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(prep.main([]),0)
        self.assertFalse(json.loads(output.getvalue())['execution'])

    def test_test_and_duplicate_split_refused(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            prep.parse_args(['--splits','test'])
        with self.assertRaises(ValueError):prep.parse_args(['--splits','train','train'])

    def test_reader_hash_rejected_before_import(self):
        bad=self.root/'bad.py';bad.write_text("raise RuntimeError('must not import')")
        with self.assertRaisesRegex(ValueError,'SHA256 differs'):prep.load_reader(bad)

    def test_native_values_actual_horizon_source_order_and_keys(self):
        first,positions,types=payload(frames=8,key=9)
        second,_,_=payload(frames=11,offset=.01,key=2)
        records,summary,source,row=self.convert([first,second])
        self.assertEqual(summary['record_count'],2)
        self.assertEqual(summary['dimension'],3)
        self.assertEqual(summary['particle_count_min'],3)
        self.assertEqual(summary['particle_count_max'],3)
        self.assertEqual(summary['particle_counts_sorted'],[3,3])
        self.assertEqual(summary['eligible_six_frame_histories'],7)
        self.assertEqual(summary['frame_lengths'],[8,11])
        self.assertEqual(summary['forecast_horizons_after_six_frames'],[2,5])
        self.assertFalse(summary['all_match_official_parser_frame_count'])
        self.assertEqual([r['source_key'] for r in records],[[9,109],[2,102]])
        self.assertEqual([r['source_index'] for r in records],[0,1])
        for field,expected in [('positions',positions),('particle_types',types)]:
            saved=self.output/'.train.staging'/Path(records[0][field]['path']).name
            array=np.load(saved,allow_pickle=False)
            self.assertEqual(array.dtype,expected.dtype)
            self.assertEqual(array.shape,expected.shape)
            self.assertEqual(array.tobytes(),expected.tobytes())
            self.assertEqual(prep.sha(saved),records[0][field]['sha256'])
        self.assertEqual(prep.sha(source),row['sha256'])
        self.assertEqual(summary['particle_type_ids'],[0,3,8])

    def test_auxiliary_nan_payload_bits_are_preserved_and_not_admitted(self):
        raw,_,_=payload(auxiliary=True)
        records,summary,_,_=self.convert([raw])
        self.assertTrue(summary['any_auxiliary_fields'])
        self.assertFalse(summary['records'][0]['auxiliary_fields'][0]['all_finite'])
        self.assertIn('unreviewed',records[0]['step_context']['use'])
        saved=np.load(self.output/'.train.staging'/Path(records[0]['step_context']['path']).name,allow_pickle=False)
        self.assertEqual(saved[0].tobytes(),np.array([0x7fc12345,0x80000000],dtype='<u4').tobytes())

    def test_two_dimensional_payload_is_not_reinterpreted_as_3d(self):
        raw,_,_=payload()
        example=reader.example_pb2.SequenceExample();example.ParseFromString(raw)
        for frame in example.feature_lists.feature_list['position'].feature:
            frame.bytes_list.value[0]=np.zeros((3,2),dtype='<f4').tobytes()
        with self.assertRaisesRegex(reader.TFRecordIntegrityError,'Position size mismatch'):
            self.convert([example.SerializeToString()])
        self.assertTrue((self.output/'.train.staging/failure.json').exists())

    def test_complete_record_crc_corruption_retains_failure(self):
        raw,_,_=payload()
        source=self.root/'train.tfrecord';row=write_records(source,[raw])
        data=bytearray(source.read_bytes());data[-1]^=1;source.write_bytes(data)
        row.update(sha256=prep.sha(source),crc32c_base64=base64.b64encode(reader.crc32c.crc32c(data).to_bytes(4,'big')).decode())
        with self.assertRaises(reader.TFRecordIntegrityError):
            prep.convert_one(reader,source,'train',row,self.output,self.metadata,set())
        failure=json.loads((self.output/'.train.staging/failure.json').read_text())
        self.assertIn('CRC',failure['reason'])
        self.assertTrue(source.exists())

    def test_whole_object_checksum_failure_before_staging(self):
        raw,_,_=payload();source=self.root/'train.tfrecord';row=write_records(source,[raw])
        for field in ('sha256','crc32c_base64','received_bytes'):
            bad={**row,field:0 if field=='received_bytes' else 'wrong'}
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'length/SHA256/CRC32C'):
                prep.convert_one(reader,source,'train',bad,self.output,self.metadata,set())
        self.assertFalse((self.output/'.train.staging').exists())

    def test_source_crc_and_sha_continue_across_multiple_hashing_chunks(self):
        # The official train file spans many 1 MiB blocks. Exercise incremental
        # CRC state with synthetic bytes without decoding or acquiring a source.
        raw = bytes(range(256)) * 8193
        source = self.root/'multichunk.bin'
        source.write_bytes(raw)
        row = {'received_bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest(),
               'crc32c_base64':base64.b64encode(reader.crc32c.crc32c(raw).to_bytes(4,'big')).decode()}
        self.assertEqual(prep.verify_source(reader,source,row).st_size,len(raw))
        source.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError,'length/SHA256/CRC32C'):
            prep.verify_source(reader,source,row)

    def test_duplicate_within_and_across_splits_not_silently_accepted(self):
        raw,_,_=payload();different_key,_,_=payload(key=99)
        with self.assertRaisesRegex(ValueError,'duplicate'):self.convert([raw,different_key])
        self.assertEqual(json.loads((self.output/'.train.staging/failure.json').read_text())['completed_records'],1)
        second_output=self.root/'second';second_output.mkdir()
        self.output=second_output;seen=set()
        self.convert([raw],seen=seen)
        with self.assertRaisesRegex(ValueError,'duplicate'):self.convert([raw],split='valid',seen=seen)

    def test_unexpected_fields_and_nonfinite_positions_preserved_as_failure(self):
        for which in ('unknown','nan','short'):
            raw,_,_=payload(frames=6 if which=='short' else 9)
            x=reader.example_pb2.SequenceExample();x.ParseFromString(raw)
            if which=='unknown':x.context.feature['extra'].int64_list.value.append(5)
            if which=='nan':x.feature_lists.feature_list['position'].feature[0].bytes_list.value[0]=np.full((3,3),np.nan,dtype='<f4').tobytes()
            out=self.root/which;out.mkdir()
            source=self.root/(which+'.tfrecord');row=write_records(source,[x.SerializeToString()])
            with self.subTest(which=which),self.assertRaises(Exception):prep.convert_one(reader,source,'train',row,out,self.metadata,set())
            self.assertTrue((out/'.train.staging/failure.json').exists())

    def receipt(self):
        rows=[]
        for name,expected in prep.SOURCES.items():
            rows.append({'name':name,'saved_name':name,'url':prep.BASE_URL+name,'status':'complete',
                         'received_bytes':expected['bytes'],'generation':expected['generation'],'crc32c_base64':expected['crc32c'],
                         'crc32c_verified':True,'sha256':prep.METADATA_SHA if name=='metadata.json' else '0'*64})
        return {'schema':prep.RECEIPT_SCHEMA,'status':'complete','dataset':'Goop-3D','source_family':'official_gns_tfrecord','files':rows}

    def test_receipt_exact_source_family_bytes_generation_crc_and_metadata(self):
        receipt=self.receipt();prep.check_receipt(receipt,['train','valid'])
        for field,bad in [('generation','new'),('crc32c_base64','bad'),('received_bytes',1),('url',prep.BASE_URL+'test.tfrecord'),('status','failed'),('crc32c_verified',False)]:
            changed=copy.deepcopy(receipt);changed['files'][1][field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):prep.check_receipt(changed,['train','valid'])
        changed=copy.deepcopy(receipt);changed['files'][0]['sha256']='0'*64
        with self.assertRaises(ValueError):prep.check_receipt(changed,['train','valid'])
        changed=copy.deepcopy(receipt);changed['files'].append({**changed['files'][1],'name':'test.tfrecord'})
        with self.assertRaises(ValueError):prep.check_receipt(changed,['train','valid'])

    def test_saved_byte_verification_detects_decoder_output_change(self):
        raw,_,_=payload();staging=self.root/'staging';staging.mkdir()
        record=reader.decode_record(raw,0,0,'train',staging,3,None)
        path=staging/Path(record['positions']['path']).name
        array=np.load(path,allow_pickle=False);array[0,0,1]+=1;np.save(path,array,allow_pickle=False)
        with self.assertRaisesRegex(ValueError,'Position element bytes changed'):prep.verify_arrays(reader,raw,record,staging)

    def test_existing_output_refused_without_reading_acquisition(self):
        argv=['--execute','--input-dir',str(self.root/'input'),'--output-dir',str(self.output),
              '--acquisition-report',str(self.root/'missing'),'--reader',str(READER_PATH)]
        with self.assertRaises(FileExistsError):prep.main(argv)
        self.assertFalse(list(self.output.iterdir()))

    def main_fixture(self, duplicate=False):
        source=self.root/'input';source.mkdir()
        metadata=json.dumps(self.metadata).encode().ljust(471,b' ')
        (source/'metadata.json').write_bytes(metadata)
        expected=copy.deepcopy(prep.SOURCES)
        rows=[]
        for name in expected:
            if name=='metadata.json':
                raw=metadata
                row={'received_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                     'crc32c_base64':base64.b64encode(reader.crc32c.crc32c(raw).to_bytes(4,'big')).decode()}
            else:
                raw,_,_=payload(frames=8,offset=0 if name=='train.tfrecord' or duplicate else .01)
                row=write_records(source/name,[raw])
            expected[name].update(bytes=row['received_bytes'],crc32c=row['crc32c_base64'])
            rows.append({**row,'name':name,'saved_name':name,'url':prep.BASE_URL+name,'status':'complete',
                         'generation':expected[name]['generation'],'crc32c_verified':True})
        receipt=self.root/'receipt.json'
        receipt.write_text(json.dumps({'schema':prep.RECEIPT_SCHEMA,'status':'complete','dataset':'Goop-3D',
                                      'source_family':'official_gns_tfrecord','files':rows}))
        argv=['--execute','--input-dir',str(source),'--output-dir',str(self.root/'converted'),
              '--acquisition-report',str(receipt),'--reader',str(READER_PATH)]
        return argv,{'SOURCES':expected,'METADATA_SHA':hashlib.sha256(metadata).hexdigest()}

    def test_main_publishes_both_only_after_complete_exact_conversion(self):
        argv,patches=self.main_fixture()
        with mock.patch.multiple(prep,**patches),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prep.main(argv),0)
        output=self.root/'converted'
        report=json.loads((output/'structural_report.json').read_text())
        self.assertEqual(report['status'],'complete_structural_only')
        self.assertFalse(report['scientific_training_admission'])
        self.assertFalse(report['test_accessed'])
        for split in ('train','valid'):
            path=output/(split+'.json');manifest=json.loads(path.read_text())
            self.assertEqual(manifest['dataset'],'Goop-3D')
            self.assertEqual(manifest['source']['family'],'official_gns_tfrecord')
            self.assertEqual(manifest['record_count'],1)
            self.assertEqual(manifest['records'][0]['positions']['shape'][0],8)
            self.assertEqual(report['splits'][split]['manifest_sha256'],prep.sha(path))
            self.assertFalse((output/('.'+split+'.staging')).exists())
            self.assertNotIn('evaluation_reservation',manifest)

    def test_main_duplicate_valid_retains_train_staging_no_manifest_publication(self):
        argv,patches=self.main_fixture(duplicate=True)
        with mock.patch.multiple(prep,**patches),self.assertRaisesRegex(ValueError,'duplicate'):
            prep.main(argv)
        output=self.root/'converted'
        self.assertEqual(json.loads((output/'structural_report.json').read_text())['status'],'failed')
        self.assertTrue((output/'.train.staging/position_000000.npy').exists())
        self.assertTrue((output/'.valid.staging/failure.json').exists())
        self.assertFalse((output/'train.json').exists())
        self.assertFalse((output/'valid.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
