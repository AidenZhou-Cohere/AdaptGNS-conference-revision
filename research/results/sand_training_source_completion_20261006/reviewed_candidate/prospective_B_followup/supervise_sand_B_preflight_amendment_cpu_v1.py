#!/usr/bin/env python3
"""One frozen Sand preflight worker in a separately human-approved B-only600s amendment.

Description only unless --execute is supplied with a root-issued operational
release. No model, array, benchmark, or frozen scientific module is imported.
The original tool exit and timeout/supervisor/worker native closure still require
root verification; this program cannot certify its own later disappearance.
"""
import time
ENTRY_MONOTONIC = time.monotonic()
import ctypes
import os
import signal
import sys


class SigEvent(ctypes.Structure):
    # Linux glibc LP64: sigval_t8, signo4, notify4, union padding48.
    _fields_=[('value',ctypes.c_void_p),('signo',ctypes.c_int),('notify',ctypes.c_int),('padding',ctypes.c_byte*48)]
class TimeSpec(ctypes.Structure):
    _fields_=[('seconds',ctypes.c_long),('nanoseconds',ctypes.c_long)]
class ITimerSpec(ctypes.Structure):
    _fields_=[('interval',TimeSpec),('value',TimeSpec)]


def arm_absolute_guard(deadline_ns, libc=None):
    """Kernel SIGKILL timer, independent of Python/worker SIGALRM handlers.

    Never delete this timer: the kernel removes it only when this owner exits.
    Literal timing is checked against the hash-bound release after arming.
    """
    if not (sys.platform.startswith('linux') and ctypes.sizeof(ctypes.c_long)==8
            and ctypes.sizeof(ctypes.c_void_p)==8 and ctypes.sizeof(SigEvent)==64
            and ctypes.sizeof(TimeSpec)==16 and ctypes.sizeof(ITimerSpec)==32):
        raise ValueError('Linux LP64 native timer ABI required')
    if not (type(deadline_ns) is int and 0<deadline_ns<2**63):
        raise ValueError('Positive finite signed64-bit absolute deadline required')
    current=time.monotonic_ns()
    if not 0<deadline_ns-current<=3600*10**9:
        raise ValueError('Original absolute deadline expired or exceeds one hour')
    lib=libc if libc is not None else ctypes.CDLL(None,use_errno=True)
    lib.timer_create.argtypes=[ctypes.c_int,ctypes.POINTER(SigEvent),ctypes.POINTER(ctypes.c_void_p)]
    lib.timer_create.restype=ctypes.c_int
    lib.timer_settime.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ITimerSpec),ctypes.c_void_p]
    lib.timer_settime.restype=ctypes.c_int
    event=SigEvent();event.signo=signal.SIGKILL;event.notify=0  # SIGEV_SIGNAL
    timer=ctypes.c_void_p()
    if lib.timer_create(time.CLOCK_MONOTONIC,ctypes.byref(event),ctypes.byref(timer))!=0:
        raise OSError(ctypes.get_errno(),'timer_create failed')
    when=ITimerSpec();when.value.seconds,when.value.nanoseconds=divmod(deadline_ns,10**9)
    if lib.timer_settime(timer,1,ctypes.byref(when),None)!=0:  # TIMER_ABSTIME
        raise OSError(ctypes.get_errno(),'absolute timer_settime failed')
    return dict(clock='CLOCK_MONOTONIC',signal='SIGKILL',absolute_deadline_ns=deadline_ns,
                armed_monotonic_ns=time.monotonic_ns(),timer_id=timer.value or 0,deleted_before_exit=False)


NATIVE_GUARD=None
if __name__=='__main__' and '--execute' in sys.argv:
    # No release, scientific source or output file has been opened at this point.
    if sys.argv.count('--hard-deadline-monotonic-ns')!=1:
        raise ValueError('One literal absolute native deadline required')
    index=sys.argv.index('--hard-deadline-monotonic-ns')
    NATIVE_GUARD=arm_absolute_guard(int(sys.argv[index+1]))
    if sys.argv.count('--outer-computed-monotonic-ns')!=1:
        raise ValueError('Literal original bootstrap computation time required')
    computed=int(sys.argv[sys.argv.index('--outer-computed-monotonic-ns')+1])
    if not 0<=NATIVE_GUARD['armed_monotonic_ns']-computed<=5*10**9:
        raise ValueError('Explicit5-second bootstrap allowance exceeded; guard remains armed')

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess

SCHEMA = 'adaptgns_sand_B_preflight_amendment_cpu_operational_release_v1'
REMOTE = '/root/repos/AdaptGNS-cuda-20261006'
PREP = REMOTE + '/cuda_preparation'
PYTHON = REMOTE + '/.venv/bin/python'
PYTHON_REAL = '/usr/bin/python3.12'
PYTHON_SHA = '6242e0e8650d7dbdebbc25e08bf4c9359ddaf65f54fcae0e57fa99395fa5357a'
VENV_SHA = 'b7d360e62970717794ce0ab4f1e1398a49a8a73a138d34ca920b1eb251fe2f3b'
COMMANDS_SHA = '9d05860204929791306485ef99e1b4e1fb72423dbad631e025de9252e5ce2e13'
GLOBAL_STOP = datetime(2026, 10, 7, 4, tzinfo=timezone.utc)
RECOVERY_SHA = '194437c62c992a061e2b3e4f3f7d12d487d7edcca2a2c701e185d3d05d3bd9ca'
COHORT = 'sand_graph_support_scoped_100k_20261006_v1'
VERIFY_PHASE_SHA='61df824a2bb23d5270f07682ba01ba3028d5019997a9cf6ba47c82ed54241914'
PREPARER_SHA='6a40fe9196157c0666ffe556eb0e92781e9fc3b84f0c45b5ff67e09be0fce1b8'
OPERATIONS={'preflight_valid':('preflight','valid'),'preflight_test':('preflight','test')}
COMMON_FLAGS=('--cohort','--cohort-audit','--root-release','--output-dir')
SPLIT_FLAGS=('--train-manifest','--valid-manifest','--test-manifest','--census-report',
            '--structural-report','--protocol','--train-admission')
MODE_FLAGS={'preflight':SPLIT_FLAGS+('--admission',)}
SOURCES = {
    'prepare_sand_recovered_final_cohort_v1.py':'41bef815edf151fddd9ef9d667a339b5cb0f46ae405d8e4ed6bf00df69e75ced',
    'prepare_sand_final_cohort_scoped_v1.py':'ebad4e30edbdb9f444ed07dabf50b31540ecf7f85b134ee059c150d1ff68ca5f',
    'audit_sand_runtime_migration_endpoints_v1.py':'7fc6a6000ecb00168beeb00c695c7eb8feb25db28c6436afd7a3aceb563ea7db',
    'supervise_sand_runtime_migration_recovery_v2.py':'58625b827484b46b9c3d852bc4d71d16e48766b1c416050cc8c3dc5640bd21fe',
    'train_sand_runtime_migration_recovery_v1.py':'0faf1bb8ff25ac1487506781dba9722361682c725eeb7915874c90f3b66499e6',
    'train_sand_graph_support_cuda.py':'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124',
    'benchmark_sand_graph_support_rollout.py':'8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13',
    'evaluate_sand_graph_support_final.py':'952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58',
    'sand_graph_support_100k_protocol_v1.md':'e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d',
    'sand_train_admission.json':'fbc8ddd94d90adea6e8f4b07caa73931dfcea41e86c8f2f830e3f01f6a177b73',
    'prepare_sand_reserved_test_recovered_v1.py':PREPARER_SHA,
    'download_sand_public_v2.py':'864d9c974deabceb19b5f3b8e88ced1ea7281595a0d3a526e8d39f382aa4605a',
    'repackage_designsafe_sand.py':'37c135a9eaa484a5fdbed2176780913b71cd3be4df602983fcd3b644c1a57ff9',
    'sand_scoped_operational_amendment_v1.md':'411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738',
    'sand_scoped_schedule_fixed_spec_v2.json':'403a96c1304341ea1899c994e0ab7474280add44ea5986ec41570e08d16c337b'}
COHORT_ADAPTER_SHA=SOURCES['prepare_sand_recovered_final_cohort_v1.py']
PROTOCOL_SHA=SOURCES['sand_graph_support_100k_protocol_v1.md']
TRAIN_ADMISSION_SHA=SOURCES['sand_train_admission.json']
TRAINER_SHA=SOURCES['train_sand_graph_support_cuda.py']
AMENDMENT_SHA=SOURCES['sand_scoped_operational_amendment_v1.md']
PLAN_SHA=SOURCES['sand_scoped_schedule_fixed_spec_v2.json']
TRAIN_MANIFEST_SHA='f133a629a0d94b67875267245ad7bda67464731caf83e2411743a5a85a71bb1f'

B_HOST='aidenzhou-aquamarine-toad-75-6d8b45c98d-mgjlq'
AMENDMENT_ROOT='/root/repos/AdaptGNS-cuda-20261006/sand_reserved_B_preflight_amendment_20261006_v1'
AMENDMENT_CONTROLS='/root/repos/AdaptGNS-cuda-20261006/sand_reserved_B_preflight_amendment_20261006_v1/controls'
AMENDMENT_ID='sand_B_preflight_completion_600s_20261006_v1'
ORIGINAL_SOURCE_SHA='895e25aa38490f42bb2f24a4129ccde8c1e91f8f01a8cf960eda15d0ee15e7ed'
ORIGINAL_SOURCE_START='2026-10-06T18:20:33.342462+00:00'
ORIGINAL_SOURCE_STOP='2026-10-06T18:35:33.342462+00:00'
HISTORY_BINDINGS={'original_source_disposition': {'path': '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_B_preflight_amendment_20261006_v1/controls/history/sand_original_source_phase_incomplete_root_v1.json', 'sha256': '5b07542ec04a11f8fef58f563d693ccb20514dbcac603b17a5736f45b97a1120'}, 'independent_source_disposition': {'path': '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_B_preflight_amendment_20261006_v1/controls/history/sand_original_source_window_disposition_independent_review_code_audit_v1.json', 'sha256': 'd3c7493bc8a5995eb312fef061d8522a722007d6d4a5178eedb49c23928fe5a3'}, 'transfer_products_review': {'path': '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_B_preflight_amendment_20261006_v1/controls/history/sand_completed_transfer_independent_review_code_audit_v1.json', 'sha256': 'fbbb549456d289cf8d533f6e1a078c915ac0f1728c5e3314b92192cbdf4395c8'}, 'transfer_target_publication': {'path': '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_B_preflight_amendment_20261006_v1/controls/history/target_publish.json', 'sha256': '77bc2bc1ad6d9efddf5594402f4f85249e06853e8461cf85f9ad6889dbcb1e4f'}, 'transfer_original_external_exit': {'path': '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_B_preflight_amendment_20261006_v1/controls/history/root_original_external_exit.json', 'sha256': 'b89c1844e4cae32b191da7c0e2300c7351833b704e68fa8cd7313760fa847f3f'}}
TRANSFER_FILES={'/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/audit_sand_runtime_migration_endpoints_v1.py': '7fc6a6000ecb00168beeb00c695c7eb8feb25db28c6436afd7a3aceb563ea7db', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py': '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/download_sand_public_v2.py': '864d9c974deabceb19b5f3b8e88ced1ea7281595a0d3a526e8d39f382aa4605a', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py': '952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/launch_sand_reserved_source_cpu_v1.py': '8b3cc0cb47edc7d019cb55c733a481ab916e3b4a8e59567122119ca023b34ed0', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/prepare_sand_final_cohort_scoped_v1.py': 'ebad4e30edbdb9f444ed07dabf50b31540ecf7f85b134ee059c150d1ff68ca5f', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/prepare_sand_recovered_final_cohort_v1.py': '41bef815edf151fddd9ef9d667a339b5cb0f46ae405d8e4ed6bf00df69e75ced', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/prepare_sand_reserved_test_recovered_v1.py': '6a40fe9196157c0666ffe556eb0e92781e9fc3b84f0c45b5ff67e09be0fce1b8', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/repackage_designsafe_sand.py': '37c135a9eaa484a5fdbed2176780913b71cd3be4df602983fcd3b644c1a57ff9', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md': 'e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json': 'bc63fabfc07a663ab866f454c07e984dd57b8d4b32aa06a760838c4f84da72e6', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_original_reserved_source_phase_20261006_v1.json': '895e25aa38490f42bb2f24a4129ccde8c1e91f8f01a8cf960eda15d0ee15e7ed', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_post_training_verification_phase_20261006_v1.json': '61df824a2bb23d5270f07682ba01ba3028d5019997a9cf6ba47c82ed54241914', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_scoped_operational_amendment_v1.md': '411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_scoped_schedule_fixed_spec_v2.json': '403a96c1304341ea1899c994e0ab7474280add44ea5986ec41570e08d16c337b', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json': 'fbc8ddd94d90adea6e8f4b07caa73931dfcea41e86c8f2f830e3f01f6a177b73', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_reserved_source_cpu_v1.py': 'd7880125e43ea6eeb047e7b3170a06b2ff1b189b9051978234adcf4d2b297955', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_runtime_migration_recovery_v2.py': '58625b827484b46b9c3d852bc4d71d16e48766b1c416050cc8c3dc5640bd21fe', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py': 'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_runtime_migration_recovery_v1.py': '0faf1bb8ff25ac1487506781dba9722361682c725eeb7915874c90f3b66499e6', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/metadata.json': 'cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/train.json': 'f133a629a0d94b67875267245ad7bda67464731caf83e2411743a5a85a71bb1f', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json': '7c61b7b4633287fe18c74c94fa56f84aa9ce06e9f23002c853eea3a12d8671f4', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json': '0e73e272a4ff386ca1e89bd010c476a82f993ced8f32176144f53b779f32d8ba', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json': '0786ee20abc2333c3bc2b1536f42f5b2f949617a331e70bb7897e9f4ee0e7d37', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/root_release.json': '68ff9d0aafa40f8681cf0018a8ad7541439c4ade67f4af2b4a6a171d297ad69e', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/acquired/download_report.json': '93b3acdd893a8dfbf1371c49d835b14762d70f9f429a75c657d4024a2e9919ff', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/acquired/metadata.json': 'cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/acquired/root_preparation_receipt.json': '32883d72dee52e35b607cc3b7a10e111ed056caeab4f7209de05a79a57361fb8', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/candidate_test/split_admission_candidate.json': 'a517f40f82e53c06eebee699ccb96512ee57343ecdde92747f7e4ae4391eadd2', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/candidate_valid/split_admission_candidate.json': 'e8100a60f5458a1250728a380bb380a72f7610b67034fff8c00da352168b59c1', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/census/all_split_census.json': 'd5de93fdd581d0cba192c4db45a9b5dbde48fe8d64e4fa674ed0c3de7e47c002', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/cohort_build_external_closure.json': '391816259ddc2153247558090880208ff0e0abb1f2999e098ba536edde04fc27', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json': '6d6881cfa45d1de9876622862034be63329c33e341f85aba30902f56c4105153', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json': 'dd6cd2590b6c732256c222129ac1e3b1677bfa00601c56fa897fb68ef510a5db', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/metadata.json': 'cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json': 'ec33156cc4fb126a42330248a70098a6f9c04dfcfdc795799472b703ec788d2d', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json': '53bed919f0f65a6308ff25388d0a43c864ded00f28b73cbfe1291e6b1c989e1c', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000000.npy': '38012c42eef87fd804098ed813180ce6ec728f2899f39c07dedb1870992d8b04', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000001.npy': 'de8d11aeaf6adade819f065bbbd4935818c4896ae40df0455cae2beb0f322bab', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000002.npy': '6711169c5f89c909f119d17ed6e107da3ae35f7880eadd32247c9346a47cd556', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000003.npy': '0c332478bd504a02037e864f9f18b8c9b297c99f58c8d9b13a88573ce89b5f22', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000004.npy': '988d9cceac00f379cf49de52ed79e506e3639a6729513ad57e970c22b4e60fd9', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000005.npy': '40305aab0a36fb8c34bfda5536b8e26c9d088e0ed6108e92649994169d80bd19', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000006.npy': '79466fd9e704c1a4dde07b787222654748e1803aa6e728b94220cb1306f4f5bd', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000007.npy': '4e766c29e4b78118100f9d17bae651aa142d0d3af73af0ab232e7e0a07569e34', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000008.npy': '387d2b7833042aaa9699dde0f121fd4794724472ad0b216067ca518166ae9ceb', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000009.npy': '9b7b2f7b8e65cd2be25ce5d1263354ea1ee7a20703e2e7bb8f19c0874f0f868c', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000010.npy': '6017cac7e806f040870bcf69e70b2103ba5b2ffddcbd4cec1f274bddcecabcf4', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000011.npy': '882fdd324d2ef6bb3cefcb03cccd8c62290def7489ea78308bcaa29d83f71e78', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000012.npy': '37e4b950a3b392e912cbc65fb1cf0524b9270e67e2c37af01c1d232135360e42', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000013.npy': '2d9a1b0b3c194bb26dd348eb05eee102eb0de43af49c4b361968036f322ed955', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000014.npy': '9810e550d7459bf513b16ef3bee61e8ca13fe607190c2070962cf5f55beb9be5', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000015.npy': '074a649220126c9b7b6c933469aded259af36978f59dcc991905b0e1a9b443eb', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000016.npy': '5eb974945906fbaf74cfe0b12207f99076fce5b1e8d1506c0caf66bb56fa3a50', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000017.npy': 'cebcfd33420ed604fb9ef92c853075697766f8b7386cbf76ee87ab932a256aba', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000018.npy': 'b9e164ba0ed8ecf1ca9788c3769e3329e298fe799ac885438ad9d5149e36c5d6', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000019.npy': '41d2d3c857c644c0b7c34d5e3de217d3ffc1ab94bb8af6447386df603f13003c', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000020.npy': 'dd4634c255b18999d56decf0b6cd983b3631a473f77c601922f88f73531cac3b', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000021.npy': 'b717308dd69f0d697491360caeeb30fecff7674c0f0a10a7d0f6c339a2bdb169', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000022.npy': '4ade6fa4a8d1c1f0f6ee01a584957084234649576a18ad022bd742925f195fef', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000023.npy': '730a6f3cecde10a786a91f9f071211c0018268b0bdf3a1ea11a897c451dccc43', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000024.npy': 'fb44a1bbb91255f9e7695972dc62738dbc9d7af59468be9b7077454f28727d35', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000025.npy': '980478cbf4b652b72b733e637e01559805ef6754b3f82afb958f69d72c81e78f', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000026.npy': '9ea12d53416d8845e1dd439676dd5b205a56dbe65b90a264bcf8729da84c6953', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000027.npy': 'a166a5dc13cee5fa6327596b214353212efa1d24200d0d7874809ecdf9f4b98f', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000028.npy': '1bde9ac3e74fb2554021ee8f068b02589c7d3b23b6b3dce624572557d7d370c4', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/position_000029.npy': '8842ac57b17ffae56d2f22569b1aeaaf14e031c2c4eacc38ffba9056f4550bba', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000000.npy': '2060e41309b3e75fb49932e8386a5c667460e36e834eb79a9950e6d0ceca3167', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000001.npy': '1de5b2aa040bb8a456ac937181fdf283be86162bc803f583aa60a7d5b419bb9f', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000002.npy': 'ba88d49128263e35ed9a78f09e718c6e4f5b573e2388020df670e69d29e468c0', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000003.npy': '8050f578977c4354df93c677e47f9f0f69871e74ab97641bc7212240074f8a73', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000004.npy': '47d4b91d3b6ac0857fe670c7fa8c24cc6668d8a700858a3666bb4ef9b423763f', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000005.npy': 'c29f89b81f567d738f5e92c68871e65917f1160f039aaec466ba47e6129f0dda', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000006.npy': '1067368ab071d524be6c6e9f8c47f519b413d68dce9cc273887a91b76b046e42', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000007.npy': '7d52ceca77d7caec8d2bf3c0a608994b3ae453a522eae6a863ac9b980a895977', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000008.npy': '8d206bda4f41768e1135eb2d877e585ef933735e58a3b765802f532eafdd332b', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000009.npy': 'af8803429a738d2ecc36bf07cb9c04cf403479c3c10d518ccdcbd6eb1f370dd5', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000010.npy': '6502eb7f2de1f7cebd62ef5cd392f384caacf29257f04bd193e1a23350285026', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000011.npy': 'e39b4af97181ae9d9acd1bb94540513090380091f74c14e7c7c3c29dfc06b819', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000012.npy': '1a22903a81aeeaa5432f83818799778e779d10919a4dd116d959938044307cb2', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000013.npy': '0e37f56b72cb6c85fd5eb711483c8c05e951b90bf1136c93a34a02fd0d8f45ee', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000014.npy': '587d2bcf516356e7dac873df084f5a5bd9c1353d7a3ed9d3c01f9693a369dbe5', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000015.npy': '8026da2400f82f89b5fae9066355dc226de7dfecc1caacd1c848eaaac5718b89', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000016.npy': '171ca09d386ad536d76484dd627e8d7b51347ce217882c5a7e868af33fcbf8c7', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000017.npy': 'e18dd7fd7cee384ab3fa84d0f15f8da864d67c571c618bab0f160a48c41980bd', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000018.npy': '92a8b93ec64ec70248346a418aae346cc24e6ab7e52802a38e9bf948f28b9327', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000019.npy': 'f0a7785701ff7dba85c51a0da9bc616c08cc102571a3f5cb64a02f5c95222e9c', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000020.npy': 'e2f4ac9df00645ae579ddbf487b1b8d57869bcb3972a62f0c95a12c28d4de093', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000021.npy': '5d68444d3af917206d9e709a2891e89502080784139ac60e47b1ce33eb5b1cc1', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000022.npy': 'a8c4cb5d4121ad813809b3c7b6075b906b1b2141977955e500f3cf48b21d1552', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000023.npy': '00a9ba999507131e1384028e5e9c64d1a7d72ab9afd9e8372f8232253e55a892', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000024.npy': 'c569a124a1e90e78ec8f811218c9e818ad96f9a3330d7781bd5fe1ccbab06aaa', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000025.npy': 'b724ea4e2fba23898134f343c2f17a6862ace483f7c9df5a7c7921b198b840c3', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000026.npy': 'e7567166b3cb25218767455482bff60071d3a22920f4ac6c3f802a10d44d6e19', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000027.npy': 'dcfea2dd3b7cae7f70fa1a8ba5e1eb9c4540e7561e8b4543948cbbb0e1c82077', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000028.npy': 'f44c0bb3e0877ade9ba312b76405e6991e0ecd025a867df4d3102afc2b82199a', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test/type_000029.npy': 'd630561030465e7e147cb1dc973a626f4739d86d318a9edd58aebf6419c74bd4'}


def require(ok, message):
    if not ok: raise ValueError(message)


def utc(): return datetime.now(timezone.utc)
def digest(v): return isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
def stamp(v):
    t = datetime.fromisoformat(v.replace('Z', '+00:00'))
    require(t.tzinfo is not None, 'Timezone required')
    return t.astimezone(timezone.utc)
def encoded(v): return (json.dumps(v, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def canonical(v):
    p = Path(v)
    require(p.is_absolute() and str(p.resolve()) == str(p), 'Canonical absolute path required')
    return p


def sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            check(); block = f.read(1 << 20); check()
            if not block: return h.hexdigest()
            h.update(block)


def read_json(binding, check=lambda: None):
    p = canonical(binding['path'])
    require(p.is_file() and not p.is_symlink() and p.stat().st_size <= 16*2**20, 'Bound ordinary JSON <=16 MiB required')
    check(); raw = p.read_bytes(); check()
    require(digest(binding['sha256']) and hashlib.sha256(raw).hexdigest() == binding['sha256'], 'Control bytes changed')
    return json.loads(raw)


def publish(path, value, check, verify_after_encode=lambda: None):
    check(); raw = encoded(value); check()
    verify_after_encode(); check()
    with Path(path).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    check()
    return hashlib.sha256(raw).hexdigest()


def native(pid):
    root = Path('/proc')/str(pid)
    try:
        a = (root/'stat').read_text().rsplit(') ', 1)[1].split()
        argv = [p.decode(errors='surrogateescape') for p in (root/'cmdline').read_bytes().split(b'\0') if p]
        try: exe = os.readlink(root/'exe')
        except FileNotFoundError: exe = None
        b = (root/'stat').read_text().rsplit(') ', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError): return None
    require(a[19] == b[19] and a[1:4] == b[1:4], 'Native process identity changed during read')
    return dict(pid=pid, ppid=int(b[1]), pgid=int(b[2]), sid=int(b[3]), start_id=b[19],
                state=b[0], argv=argv, executable=exe)


def stable(row): return {k: row[k] for k in ('pid', 'ppid', 'pgid', 'sid', 'start_id', 'argv', 'executable')}


def parent_death_hook(parent_pid):
    # Popen is single threaded. Setting PDEATHSIG before the second parent check
    # closes the parent-exit race; exec retains the setting for this interpreter.
    def hook():
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0: os._exit(126)
        if os.getppid() != parent_pid: os._exit(126)
    return hook


class Runtime:
    mono = staticmethod(time.monotonic)
    now = staticmethod(utc)
    sleep = staticmethod(time.sleep)
    identity = staticmethod(native)
    def boot(self): return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    def bootstrap_start_bounds(self,row):
        # /proc start ticks use BOOTTIME. Measure both sides of MONOTONIC to
        # bound the offset rather than assuming the two domains are equal.
        boot_before=time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        mono=time.monotonic_ns()
        boot_after=time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        hz=os.sysconf('SC_CLK_TCK');ticks=int(row['start_id'])
        lower=ticks*10**9//hz-(boot_after-mono)
        upper=((ticks+1)*10**9+hz-1)//hz-(boot_before-mono)
        return dict(lower_ns=lower,upper_ns=upper,ticks=ticks,ticks_per_second=hz,
                    offset_lower_ns=boot_before-mono,offset_upper_ns=boot_after-mono)
    def open_pidfd(self, pid): return os.pidfd_open(pid, 0)
    def close_pidfd(self, fd): os.close(fd)
    def launch(self, argv, stdout, stderr):
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''; env['PYTHONDONTWRITEBYTECODE'] = '1'
        env.update({k:'2' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            env=env, close_fds=True, start_new_session=False, preexec_fn=parent_death_hook(os.getpid()))
    def reap(self, c):
        if c['reaped']: return True
        pid, status, usage = os.wait4(c['pid'], os.WNOHANG)
        if not pid: return False
        require(pid == c['pid'], 'Wrong child reaped')
        code = os.waitstatus_to_exitcode(status); c['process'].returncode = code
        c.update(reaped=True, exit_code=code, wait_status=status, reaped_utc=self.now().isoformat(),
                 reaped_monotonic=self.mono(), usage=dict(user_seconds=usage.ru_utime,
                 system_seconds=usage.ru_stime, peak_rss_kib=usage.ru_maxrss))
        return True
    def send(self, c, sig): signal.pidfd_send_signal(c['pidfd'], sig, None, 0)


class Budget:
    def __init__(self, r, rt, entry):
        self.rt, self.release, self.abort = rt, r, None
        clock = r['clock_sample']; sample = stamp(clock['host_utc'])
        require(clock['host_boot_id'] == rt.boot(), 'Host boot changed')
        m = clock['host_monotonic_seconds']; age = rt.mono()-m
        require(type(m) in (int,float) and math.isfinite(m) and 0 <= age <= 300
                and abs((rt.now()-sample).total_seconds()-age) <= 5, 'Original clock is stale/inconsistent')
        require(clock['source'] == 'root_fresh_tool_and_host_clock_evidence' and clock['error_bound_seconds'] == 5
                and abs((sample-stamp(clock['root_reference_utc'])).total_seconds()) <= 5, 'Clock uncertainty not bound')
        require(stamp(r['invocation_origin_utc']) == sample, 'Per-invocation root anchor differs')
        quota = r['remaining_shared_amendment_seconds']
        self.ends = {k:min(entry+quota, m+(stamp(r[k])-sample).total_seconds()-5) for k in
                     ('work_stop_utc', 'cleanup_deadline_utc', 'publication_deadline_utc')}
        self.check()
    def check(self, key='work_stop_utc', ignore_abort=False):
        require(ignore_abort or self.abort is None, self.abort or 'Cancelled')
        require(self.rt.mono() < self.ends[key] and self.rt.now()+timedelta(seconds=5) < stamp(self.release[key]),
                'Reached '+key)
    def cap_for_actual_outer(self,earliest_kill_ns):
        end=earliest_kill_ns/10**9
        for key,tail in (('work_stop_utc',15),('cleanup_deadline_utc',5),('publication_deadline_utc',0)):
            self.ends[key]=min(self.ends[key],end-tail)
        self.check()


def structure(r):
    require(sys.platform.startswith('linux'), 'Linux only')
    require(r.get('schema') == SCHEMA and r.get('issued_by') == 'root'
            and r.get('status') == 'approved_one_bounded_B_preflight_amendment_invocation', 'Unadmitted amendment candidate')
    require(r.get('operation') in OPERATIONS and r.get('hostname') == B_HOST == socket.gethostname(),
            'Only the two original never-executed B preflights are allowed')
    require(r.get('new_or_restarted_clock_granted') is True and r.get('original_source_clock_restarted') is False
            and r.get('automatic_retry') is False and r.get('cleanup_and_publication_seconds') == 15,
            'Explicit new amendment clock must preserve original clock and no-retry policy')
    origin=stamp(r['invocation_origin_utc']);end=stamp(r['publication_deadline_utc'])
    remaining=(end-origin).total_seconds()
    require(type(r.get('remaining_shared_amendment_seconds')) in (int,float) and math.isfinite(remaining)
            and 0<remaining<=600 and r['remaining_shared_amendment_seconds']==remaining,
            'Only actual remaining time in the one shared600-second amendment is available')
    require(stamp(r['cleanup_deadline_utc'])==end-timedelta(seconds=5)
            and stamp(r['work_stop_utc'])==end-timedelta(seconds=15), 'One included15-second final tail required')
    require(origin<=stamp(r['issued_utc'])<=origin+timedelta(seconds=300),'Fresh operation issuance required')


def verify(r, path, pin, budget):
    structure(r); budget.check()
    require(sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode,'Owner must use -I -S -B')
    hard_ns=r['hard_deadline_monotonic_ns'];computed_ns=r['_outer_computed_monotonic_ns'];seconds=r['_outer_term_seconds']
    require(type(hard_ns) is int and hard_ns==int(budget.ends['publication_deadline_utc']*10**9)
            and r['bootstrap_allowance_seconds']==5 and r['_native_guard'] is not None
            and r['_native_guard']['absolute_deadline_ns']==hard_ns
            and 0<=r['_native_guard']['armed_monotonic_ns']-computed_ns<=5*10**9,
            'Literal native deadline/bootstrap must bind original root clock')
    require(type(seconds) is int and seconds>0 and type(computed_ns) is int
            and seconds==(hard_ns-computed_ns)//10**9-15-5
            and computed_ns+(seconds+15+5)*10**9<=hard_ns,'Outer duration must use actual original remaining time')
    timeout=r['outer_timeout']
    expected_outer=[timeout['path'],'--signal=TERM','--kill-after=15s',str(seconds)+'s',PYTHON,'-I','-S','-B',
        r['owner_source']['path'],'--execute','--hard-deadline-monotonic-ns',str(hard_ns),
        '--outer-term-seconds',str(seconds),'--outer-computed-monotonic-ns',str(computed_ns),
        '--release',str(path),'--release-sha256',pin]
    parent=budget.rt.identity(os.getppid());own=budget.rt.identity(os.getpid())
    require(parent is not None and parent['argv']==expected_outer and parent['executable']==str(Path(timeout['path']).resolve())
            and own is not None and own['pgid']==parent['pgid'] and own['sid']==parent['sid'], 'Exact native derived-duration outer timeout required')
    started=budget.rt.bootstrap_start_bounds(parent)
    require(started['lower_ns']<=computed_ns and started['upper_ns']<=r['_native_guard']['armed_monotonic_ns']
            and computed_ns<=r['_native_guard']['armed_monotonic_ns'], 'Native bootstrap clock-domain identity differs')
    # exec preserves the bootstrap PID/start ticks. These are NOT falsely
    # labelled as the moment GNU timeout arms its timer. Actual exec lies after
    # the literal calculation and before owner guard arming, within the5s bound.
    r['_native_outer_timing']=dict(bootstrap_process_start=started,computed_monotonic_ns=computed_ns,
        owner_guard_armed_monotonic_ns=r['_native_guard']['armed_monotonic_ns'],term_seconds=seconds,
        earliest_relative_kill_monotonic_ns=computed_ns+(seconds+15)*10**9,
        original_absolute_hard_deadline_ns=hard_ns,bootstrap_allowance_seconds=5)
    budget.cap_for_actual_outer(computed_ns+(seconds+15)*10**9)
    return verify_source_contract(r,path,pin,budget,parent,own)


def cohort_gate(cohort,audit):
    models=cohort.get('models',[]);expected={(a,s) for a in ('base','mix') for s in range(3)}
    require(cohort.get('schema')=='adaptgns_sand_graph_support_final_cohort_v1' and cohort.get('status')=='frozen_for_final_evaluation'
        and cohort.get('issued_by')=='root' and cohort.get('dataset')=='Sand' and cohort.get('updates')==100000
        and cohort.get('protocol_sha256')==PROTOCOL_SHA and cohort.get('training_admission_sha256')==TRAIN_ADMISSION_SHA
        and cohort.get('trainer_source_sha256')==TRAINER_SHA and cohort.get('adapter_sha256')==COHORT_ADAPTER_SHA
        and cohort.get('operational_amendment_sha256')==AMENDMENT_SHA and cohort.get('schedule_plan_sha256')==PLAN_SHA,
        'Complete reviewed Sand scoped cohort required before test access')
    require(len(models)==6 and {(m.get('arm'),m.get('seed')) for m in models}==expected
        and all(type(m.get('seed')) is int and m.get('objective')=='faithful' and m.get('completed_steps')==100000 for m in models)
        and all(isinstance(m.get('checkpoint_sha256'),str) and len(m['checkpoint_sha256'])==64
            and all(c in '0123456789abcdef' for c in m['checkpoint_sha256']) for m in models)
        and len({m['checkpoint_sha256'] for m in models})==6
        and cohort.get('policies')==['base','dense','random25','speed25','laggedrisk25','relative-velocity-RMS25'],
        'All six distinct fixed100k endpoints and six declared policies required')
    require(audit.get('schema')=='adaptgns_sand_graph_support_complete_cohort_audit_v1'
        and audit.get('status')=='all_six_endpoints_and_pairing_verified' and audit.get('adapter_sha256')==COHORT_ADAPTER_SHA
        and audit.get('cohort_id')==cohort.get('cohort_id') and audit.get('trainer_source_sha256')==TRAINER_SHA
        and audit.get('issued_by')=='root' and audit.get('protocol_sha256')==PROTOCOL_SHA
        and audit.get('training_admission_sha256')==TRAIN_ADMISSION_SHA,
        'Matching complete source/tensor/pairing audit required')
    rows=audit.get('models',[]);pairs=audit.get('paired_seeds',[])
    require(len(rows)==6 and {(m.get('arm'),m.get('seed')) for m in rows}==expected,'Complete endpoint audit grid required')
    for model in models:
        ev=next(m for m in rows if (m['arm'],m['seed'])==(model['arm'],model['seed']))
        require(ev.get('checkpoint_sha256')==model['checkpoint_sha256'] and ev.get('completed_steps')==ev.get('graph_history_updates')==100000
            and ev.get('checkpoint_every')==10000 and ev.get('log_every')==100 and all(ev.get(k) is True for k in
            ('all_optimizer_steps_equal_100000','all_state_and_moments_finite','source_data_protocol_verified','checkpoint_bytes_verified')),
            'Incomplete endpoint proof')
    require(len(pairs)==3 and {p.get('seed') for p in pairs}=={0,1,2} and all(type(p.get('seed')) is int and all(p.get(k) is True for k in
        ('initial_model_tensor_identity','initial_cpu_cuda_rng_identity','all_frame_noise_lr_schedules_equal','all_graph_budgets_and_rng_material_verified')) for p in pairs),
        'Complete three-seed initialization/schedule pairing required')


def source_command(r):
    mode,split=OPERATIONS[r['operation']]
    flags=COMMON_FLAGS+MODE_FLAGS[mode];paths=r['paths']
    require(set(paths)==set(flags),'Exact source-mode path flag set required')
    for value in paths.values():canonical(value)
    command=[PYTHON,'-B',PREP+'/prepare_sand_reserved_test_recovered_v1.py','--execute','--mode',mode]
    if split is not None:command+=['--split',split]
    for flag in flags:command += [flag,paths[flag]]
    return command


def source_outputs(mode,root):
    require(mode=='preflight','Only preflight output is allowed')
    return [root+'/split_preflight.json']


def numeric_bindings(paths,pins,check):
    result={}
    for split,count in (('train',1000),('valid',30),('test',30)):
        path=paths['--'+split+'-manifest']
        manifest=read_json(dict(path=path,sha256=pins[path]),check)
        require(manifest.get('dataset')=='Sand' and manifest.get('split')==split
                and manifest.get('record_count')==len(manifest.get('records',[]))==count,
                'Complete original1000/30/30 manifests required')
        for index,record in enumerate(manifest['records']):
            check()
            require(type(record.get('source_index')) is int and record['source_index']==index
                    and record.get('id')==f'{split}:{index:06d}','Exact complete manifest source order required')
            for key in ('positions','particle_types'):
                desc=record[key];relative=Path(desc['path']);p=Path(path).parent/relative
                require(not relative.is_absolute() and '..' not in relative.parts and p.is_file() and not p.is_symlink()
                        and str(p.resolve())==str(p) and p.resolve().is_relative_to(Path(path).parent)
                        and str(p) not in result and digest(desc.get('sha256'))
                        and type(desc.get('size_bytes')) is int and desc['size_bytes']>0
                        and p.stat().st_size==desc['size_bytes'],'Safe distinct original numeric file binding required')
                result[str(p)]=desc['sha256']
    require(len(result)==2120,'All2120 position/type file bindings required')
    return result


def amendment_gate(r,cohort_binding,audit_binding,check):
    require(r.get('source_phase')==dict(path=PREP+'/sand_original_reserved_source_phase_20261006_v1.json',sha256=ORIGINAL_SOURCE_SHA)
            and r.get('history')==HISTORY_BINDINGS,'Original source/history identity cannot change')
    original=read_json(r['source_phase'],check)
    require(original.get('schema')=='adaptgns_sand_original_reserved_source_phase_v1' and original.get('issued_by')=='root'
            and original.get('status')=='approved_original_shared_source_phase'
            and original.get('original_verification_phase_sha256')==VERIFY_PHASE_SHA
            and original.get('original_shared_source_seconds')==900 and original.get('original_evaluation_seconds')==11760
            and original.get('original_analysis_seconds')==3600 and original.get('new_or_restarted_clock_granted') is False
            and original.get('preparation_started_utc')==ORIGINAL_SOURCE_START and original.get('preparation_stop_utc')==ORIGINAL_SOURCE_STOP
            and original.get('cohort_sha256')==cohort_binding['sha256'] and original.get('cohort_audit_sha256')==audit_binding['sha256'],
            'Frozen original source allocation/history differs')
    history={k:read_json(v,check) for k,v in HISTORY_BINDINGS.items()}
    disposition=history['original_source_disposition']; independent=history['independent_source_disposition']
    require(disposition.get('issued_by')=='root' and disposition.get('status')=='original_source_phase_incomplete_B_preflights_unexecuted'
            and disposition.get('phase_sha256')==ORIGINAL_SOURCE_SHA and disposition.get('source_preparation_unexecuted')==2
            and disposition.get('all24_evaluation_stages')=='unadmitted_and_unexecuted' and disposition.get('clock_reset_or_extension') is False
            and independent.get('status')=='original_source_window_expired_incomplete'
            and independent.get('original_history_must_remain_incomplete') is True,
            'Original incomplete disposition must be preserved')
    publication=history['transfer_target_publication'];external=history['transfer_original_external_exit']
    require(publication.get('hostname')==B_HOST and publication.get('action')=='publish'
            and publication.get('files_sha256')==TRANSFER_FILES and len(TRANSFER_FILES)==98
            and stamp(publication['finished_utc'])<stamp(ORIGINAL_SOURCE_STOP)
            and external.get('observed_by')=='root' and external.get('original_session')==84335
            and external.get('original_tool_exit_code')==0
            and external.get('observed_utc')=='2026-10-06T18:35:55.561287+00:00'
            and stamp(external['observed_utc'])>stamp(ORIGINAL_SOURCE_STOP),
            'Exact98 transfer evidence and original late controller exit must remain explicit')
    binding=r['amendment_phase'];approval_binding=r['human_approval']
    require(binding.get('path')==AMENDMENT_CONTROLS+'/amendment_phase.json'
            and approval_binding.get('path')==AMENDMENT_CONTROLS+'/human_approval.json', 'Fresh amendment/approval controls required')
    phase=read_json(binding,check);approval=read_json(approval_binding,check)
    require(phase.get('schema')=='adaptgns_sand_B_preflight_completion_amendment_phase_v1'
            and phase.get('issued_by')=='root' and phase.get('status')=='approved_new_shared_B_preflight_amendment_phase'
            and phase.get('amendment_id')==AMENDMENT_ID and phase.get('hostname')==B_HOST
            and phase.get('operations')==list(OPERATIONS) and phase.get('new_shared_amendment_seconds')==600
            and phase.get('new_or_restarted_clock_granted') is True and phase.get('original_source_clock_restarted') is False
            and phase.get('original_source_phase')==r['source_phase'] and phase.get('history')==HISTORY_BINDINGS
            and phase.get('human_approval')==approval_binding
            and phase.get('reviewed_candidate_manifest_sha256')==r.get('reviewed_candidate_manifest_sha256') and phase.get('cohort_sha256')==cohort_binding['sha256']
            and phase.get('cohort_audit_sha256')==audit_binding['sha256'] and phase.get('automatic_retry') is False
            and phase.get('includes_both_operations_external_exits_native_closure_and_reviews') is True
            and phase.get('subsequent_gpu_control_seconds')==600 and phase.get('original_evaluation_seconds')==11760
            and phase.get('original_analysis_seconds')==3600 and phase.get('full_evaluation_stage_denominator')==24
            and phase.get('global_analysis_deadline_utc')==GLOBAL_STOP.isoformat(),
            'One explicit new B-only600-second phase with full downstream reserves required')
    start=stamp(phase['amendment_started_utc']);stop=stamp(phase['amendment_stop_utc'])
    require(stamp(external['observed_utc'])<=start,'Original transfer external exit must precede the new amendment anchor')
    require((stop-start).total_seconds()==600 and stamp(ORIGINAL_SOURCE_STOP)<start
            and start<=stamp(r['invocation_origin_utc'])<stop and stamp(r['publication_deadline_utc'])==stop
            and stop+timedelta(seconds=600+11760+3600+5)<=GLOBAL_STOP,
            'Exact600 amendment or separate600+11760+3600 downstream reserve unavailable')
    require(approval.get('schema')=='adaptgns_sand_B_preflight_explicit_human_approval_v1'
            and approval.get('recorded_by')=='root' and approval.get('status')=='explicit_human_approval_received'
            and approval.get('authorization_source')=='direct_human_message' and approval.get('amendment_id')==AMENDMENT_ID
            and type(approval.get('human_message_reference')) is str and bool(approval['human_message_reference'].strip())
            and type(approval.get('human_message_text')) is str and bool(approval['human_message_text'].strip())
            and approval.get('approved_new_shared_seconds')==600 and approval.get('hostname')==B_HOST
            and approval.get('operations')==list(OPERATIONS) and approval.get('original_source_phase_sha256')==ORIGINAL_SOURCE_SHA
            and approval.get('preserves_original_incomplete_history') is True and approval.get('automatic_retry') is False
            and approval.get('owner_source_sha256')==r['owner_source']['sha256']
            and approval.get('bootstrap_source_sha256')==r['bootstrap_source']['sha256']
            and approval.get('preparation_source_sha256')==PREPARER_SHA
            and digest(r.get('reviewed_candidate_manifest_sha256'))
            and approval.get('reviewed_candidate_manifest_sha256')==r['reviewed_candidate_manifest_sha256']
            and approval.get('subsequent_gpu_control_seconds')==600 and approval.get('original_evaluation_seconds')==11760
            and approval.get('original_analysis_seconds')==3600 and approval.get('global_analysis_deadline_utc')==GLOBAL_STOP.isoformat()
            and stamp(ORIGINAL_SOURCE_STOP)<stamp(approval['human_approval_observed_utc'])<=stamp(approval['recorded_utc'])<=start
            and start<=stamp(phase['issued_utc'])<=stamp(r['issued_utc']),
            'Actual direct human approval must precede the new amendment anchor; null/template authority is forbidden')
    expected={v['path']:v['sha256'] for v in HISTORY_BINDINGS.values()}
    expected.update({binding['path']:binding['sha256'],approval_binding['path']:approval_binding['sha256']})
    return original,phase,start,stop,expected


def verify_source_contract(r,path,pin,budget,parent,own):
    require(sys.executable==PYTHON and str(Path(sys.executable).resolve())==PYTHON_REAL
            and r['owner_source']['path']==PREP+'/supervise_sand_B_preflight_amendment_cpu_v1.py'
            and str(Path(__file__).resolve())==r['owner_source']['path'], 'Exact source owner/interpreter required')
    require(r['bootstrap_source']['path']==PREP+'/launch_sand_B_preflight_amendment_cpu_v1.py','Exact source bootstrap required')
    require(r['command']==source_command(r),'Exact allowlisted source worker argv required')
    mode,split=OPERATIONS[r['operation']];paths=r['paths']
    expected_paths={
        '--cohort':REMOTE+'/sand_post_training_A_20261006_v1/cohort/cohort.json',
        '--cohort-audit':REMOTE+'/sand_post_training_A_20261006_v1/cohort/cohort_audit.json',
        '--root-release':AMENDMENT_CONTROLS+'/'+r['operation']+'.mode_release.json',
        '--output-dir':AMENDMENT_ROOT+'/'+r['operation'],
        '--train-manifest':REMOTE+'/sand_numeric_train_valid_20261006_v1/train.json',
        '--valid-manifest':REMOTE+'/sand_numeric_train_valid_20261006_v1/valid.json',
        '--test-manifest':REMOTE+'/sand_reserved_preparation_20261006_v1/numeric/test.json',
        '--census-report':REMOTE+'/sand_reserved_preparation_20261006_v1/census/all_split_census.json',
        '--structural-report':PREP+'/sand_numeric_structural_report.json' if split=='valid' else REMOTE+'/sand_reserved_preparation_20261006_v1/numeric/structural_report.json',
        '--protocol':PREP+'/sand_graph_support_100k_protocol_v1.md', '--train-admission':PREP+'/sand_train_admission.json',
        '--admission':REMOTE+'/sand_reserved_preparation_20261006_v1/controls/'+split+'.root_admission.json'}
    require(paths==expected_paths and r['owner_output_dir']==AMENDMENT_ROOT+'/owners/'+r['operation']
            and str(path)==AMENDMENT_CONTROLS+'/'+r['operation']+'.cpu_release.json',
            'Fresh exact amendment output/control paths and unchanged numerical inputs required')
    require(r['core_release']['path']==paths['--root-release'],'Separate original mode release required')
    release=read_json(r['core_release'],budget.check)
    pins=release.get('files_sha256',{})
    require(isinstance(pins,dict) and all(digest(v) for v in pins.values()),'Exact original worker byte bindings required')
    cohort_binding=dict(path=paths['--cohort'],sha256=pins.get(paths['--cohort']))
    audit_binding=dict(path=paths['--cohort-audit'],sha256=pins.get(paths['--cohort-audit']))
    # Only root controls/cohort scalar JSON are read before this complete gate.
    cohort=read_json(cohort_binding,budget.check);audit=read_json(audit_binding,budget.check)
    cohort_gate(cohort,audit)
    require(cohort.get('cohort_audit_sha256')==audit_binding['sha256'],'Cohort/audit bytes differ')
    require(release.get('schema')=='adaptgns_sand_reserved_test_preparation_release_scoped_v1'
            and release.get('issued_by')=='root' and release.get('status')=='approved_for_'+mode
            and release.get('preparation_source_sha256')==PREPARER_SHA
            and release.get('cohort_sha256')==cohort_binding['sha256']
            and release.get('cohort_audit_sha256')==audit_binding['sha256']
            and release.get('output_dir')==paths['--output-dir'] and release.get('no_retry_or_checkpoint_selection') is True,
            'Separate frozen mode authority/cohort bindings differ')
    phase,amendment,start,stop,amendment_inputs=amendment_gate(r,cohort_binding,audit_binding,budget.check)
    require(stamp(release['preparation_started_utc'])==start and stamp(release['preparation_stop_utc'])==stop
            and release.get('clock_error_bound_seconds')==5 and release.get('mode')=='preflight'
            and release.get('amendment_phase_sha256')==r['amendment_phase']['sha256']
            and release.get('original_source_phase_sha256')==ORIGINAL_SOURCE_SHA
            and stamp(cohort['created_utc'])<=start<=stamp(release['issued_utc'])<=stamp(r['issued_utc']),
            'Frozen preflight alarm must bind the one new amendment phase')
    closure=read_json(phase['cohort_build_external_closure'],budget.check)
    require(closure.get('issued_by')=='root' and closure.get('status')=='complete_cohort_inputs_and_native_closure_verified'
            and closure.get('original_tool_exit_code')==0 and closure.get('owner_terminal_status')=='complete'
            and closure.get('native_absent') and all(x is True for x in closure['native_absent'].values())
            and closure.get('all_published_input_output_hashes_verified') is True
            and closure.get('cohort_sha256')==cohort_binding['sha256'] and closure.get('cohort_audit_sha256')==audit_binding['sha256']
            and stamp(closure['checked_utc'])<=start,'Actual stopped all-six cohort closure must precede source access')
    require(r['verification_phase']['sha256']==VERIFY_PHASE_SHA,'Original verification phase cannot be replaced')
    verification=read_json(r['verification_phase'],budget.check)
    require(verification.get('issued_by')=='root' and verification.get('original_verification_seconds')==2700
            and stamp(r['issued_utc'])>=start,'Original verification lineage differs')
    fixed_names=('prepare_sand_recovered_final_cohort_v1.py','download_sand_public_v2.py','repackage_designsafe_sand.py',
        'evaluate_sand_graph_support_final.py','benchmark_sand_graph_support_rollout.py','sand_graph_support_100k_protocol_v1.md',
        'sand_scoped_operational_amendment_v1.md','sand_scoped_schedule_fixed_spec_v2.json','prepare_sand_reserved_test_recovered_v1.py')
    required={PREP+'/'+name:SOURCES[name] for name in fixed_names}
    required.update({cohort_binding['path']:cohort_binding['sha256'],audit_binding['path']:audit_binding['sha256']})
    input_flags=[f for f in MODE_FLAGS[mode] if f!='--input-dir']
    for flag in input_flags:
        require(paths[flag] in pins,'Explicit mode input lacks original root binding: '+flag)
        required[paths[flag]]=pins[paths[flag]]
    if mode in ('census','candidate','preflight'):
        require(pins[paths['--train-manifest']]==TRAIN_MANIFEST_SHA,'Original full1000 training manifest required')
    if mode in ('candidate','preflight'):
        require(paths['--protocol']==PREP+'/sand_graph_support_100k_protocol_v1.md'
                and pins[paths['--protocol']]==PROTOCOL_SHA and pins[paths['--train-admission']]==TRAIN_ADMISSION_SHA,
                'Original protocol/admission required')
    require(pins==required and all(TRANSFER_FILES.get(p)==v for p,v in pins.items()),
            'Exact minimal preflight inputs must match original transferred bytes')
    require(r['inputs_sha256']=={**pins,r['core_release']['path']:r['core_release']['sha256']},
            'Operational and original mode bindings differ')
    numeric={}
    if mode in ('census','candidate','preflight'):
        numeric=numeric_bindings(paths,pins,budget.check)
        if mode in ('candidate','preflight'):
            census=read_json(dict(path=paths['--census-report'],sha256=pins[paths['--census-report']]),budget.check)
            require(census.get('numeric_files_sha256')==numeric,'Exact complete census numeric binding map required')
    r['_numeric_input_files_count']=len(numeric)
    expected={str(path):pin,**r['inputs_sha256'],**{PREP+'/'+name:value for name,value in SOURCES.items()},
        PYTHON:PYTHON_SHA,PYTHON_REAL:PYTHON_SHA,REMOTE+'/.venv/pyvenv.cfg':VENV_SHA}
    require(all(p not in expected or expected[p]==value for p,value in numeric.items()),'Numeric/control binding conflict')
    require(all(p not in expected or expected[p]==v for p,v in TRANSFER_FILES.items()),'Original transfer/control identity conflict')
    expected.update(numeric);expected.update(TRANSFER_FILES);expected.update(amendment_inputs)
    for key in ('verification_phase','source_phase','owner_source','bootstrap_source'):
        expected[r[key]['path']]=r[key]['sha256']
    expected[phase['cohort_build_external_closure']['path']]=phase['cohort_build_external_closure']['sha256']
    timeout=r['outer_timeout']
    require(timeout['path']=='/usr/bin/timeout' and digest(timeout['sha256'])
            and timeout['duration_rule']=='floor((original_hard_deadline_ns-actual_bootstrap_monotonic_ns)/1e9)-15-5'
            and 'argv' not in timeout,'Derived remaining-time GNU timeout required')
    expected[timeout['path']]=timeout['sha256']
    outputs=[canonical(r['owner_output_dir']),canonical(paths['--output-dir'])]
    require(outputs[0]!=outputs[1] and outputs[0] not in outputs[1].parents and outputs[1] not in outputs[0].parents,
            'Separate worker and owner output trees required')
    protected=[canonical(p) for p in expected if p!=PYTHON]
    require(all(o!=p and o not in p.parents and p not in o.parents for o in outputs for p in protected),
            'Source outputs overlap protected inputs')
    require(all(not o.exists() for o in outputs),'Fresh separate output trees required')
    # The full six-model gate above precedes all source/data byte hashing here.
    for p,value in expected.items():require(sha(p,budget.check)==value,'Source/control bytes changed: '+p)
    return expected,source_outputs(mode,paths['--output-dir']),paths['--output-dir'],parent,own


def public(c): return {k:v for k,v in c.items() if k not in ('process','pidfd')}
def register(process, rt, command, owner):
    return dict(process=process,pid=process.pid,pidfd=None,identity=None,registered=True,reaped=False,
                command=command,owner_identity=owner,signals=[],cleanup_errors=[],exit_code=None,
                started_utc=rt.now().isoformat(),started_monotonic=rt.mono())
def capture(c, rt):
    # Never poll or reap before acquiring the fd: it is the cleanup capability
    # even when the subsequent argv/native capture or evidence publication fails.
    c['pidfd'] = rt.open_pidfd(c['pid'])
    row = rt.identity(c['pid']); own = c['owner_identity']
    require(row is not None and row['state'] != 'Z' and row['ppid'] == own['pid']
            and row['pgid'] == own['pgid'] and row['sid'] == own['sid']
            and row['argv'] == c['command'] and row['executable'] == PYTHON_REAL, 'Exact owned worker identity unavailable')
    c['identity'] = row
def send(c, sig, rt):
    if rt.reap(c): return False
    require(c['pidfd'] is not None, 'No owned pidfd; outer timeout/PDEATHSIG remains required')
    row = rt.identity(c['pid'])
    if c['identity'] is not None and row is not None and row['state'] != 'Z':
        require(stable(row) == stable(c['identity']), 'Owned identity changed; refusing signal')
    rt.send(c,sig); c['signals'].append(dict(signal=signal.Signals(sig).name,utc=rt.now().isoformat(),monotonic=rt.mono()))
    return True
def cleanup(c, rt, budget):
    if c is None or c['reaped']: return
    start = rt.mono(); end = min(start+10,budget.ends['cleanup_deadline_utc']); attempted=set()
    while rt.mono() < end:
        if rt.reap(c): return
        for delay,sig in ((0,signal.SIGINT),(3,signal.SIGTERM),(6,signal.SIGKILL)):
            if rt.mono() >= start+delay and sig not in attempted:
                attempted.add(sig)
                try: send(c,sig,rt)
                except BaseException as e: c['cleanup_errors'].append(type(e).__name__+': '+str(e))
        rt.sleep(min(.05,max(0,end-rt.mono())))
    rt.reap(c)


def output_hashes(paths, root, check):
    if root is not None:
        p=canonical(root); check()
        require(p.is_dir() and not p.is_symlink(),'Source output directory missing or symlinked')
        actual=[];directories=[]
        for child in p.rglob('*'):
            check();require(not child.is_symlink(),'Source output symlink')
            if child.is_file():actual.append(str(child))
            elif child.is_dir():directories.append(str(child))
            else:raise ValueError('Unexpected non-file source output')
        expected_dirs=sorted({str(Path(x).parent) for x in paths if Path(x).parent!=p})
        require(sorted(actual)==sorted(paths) and sorted(directories)==expected_dirs,
                'Partial/failed/unexpected source output tree; preserve all files')
    result={}
    for path in paths:
        check(); p=canonical(path); require(p.is_file() and not p.is_symlink(), 'Output is missing/not ordinary')
        result[path]=sha(p,check)
    return result


def execute(r, path, pin, rt, budget):
    expected, outputs, output_root, outer, own = verify(r,path,pin,budget)
    require(signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL, 'Default SIGCHLD/no automatic reaper required')
    directory=Path(r['owner_output_dir']); budget.check(); directory.mkdir(mode=0o700,exist_ok=False)
    child=None; failure=None; evidence={}; result={}
    previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def stop(sig,frame): budget.abort=budget.abort or signal.Signals(sig).name
    for s in previous: signal.signal(s,stop)
    final_check=lambda:budget.check('publication_deadline_utc',True)
    try:
        evidence[str(directory/'owner_started.json')]=publish(directory/'owner_started.json',dict(
            owner_identity=own,outer_timeout_identity=outer,release_sha256=pin,entry_monotonic=ENTRY_MONOTONIC,
            amendment_clock_sample=r['clock_sample'],native_hard_guard=r.get('_native_guard'),
            native_outer_timing=r.get('_native_outer_timing'),scientific_admission=False),budget.check)
        with (directory/'child.stdout').open('xb') as out,(directory/'child.stderr').open('xb') as err:
            budget.check(); child=register(rt.launch(r['command'],out,err),rt,r['command'],own)
            capture(child,rt)
            evidence[str(directory/'child_registered.json')]=publish(directory/'child_registered.json',public(child),budget.check)
            while not rt.reap(child): budget.check();rt.sleep(.05)
            require(child['exit_code']==0 and child['signals']==[], 'Worker failed/interrupted')
        result=output_hashes(outputs,output_root,budget.check)
        for p,value in expected.items(): require(sha(p,budget.check)==value,'Inputs changed after worker: '+p)
    except BaseException as e: failure=dict(type=type(e).__name__,error=str(e),utc=rt.now().isoformat())
    finally:
        try: cleanup(child,rt,budget)
        except BaseException as e: failure=failure or dict(type=type(e).__name__,error=str(e))
        if child is not None and child['pidfd'] is not None:
            try: rt.close_pidfd(child['pidfd']);child['pidfd']=None
            except BaseException as e: failure=failure or dict(type=type(e).__name__,error=str(e))
    try:
        final_check()
        absent=child is None or child['reaped'] and rt.identity(child['pid']) is None
        require(absent,'Worker native closure incomplete')
        for name in ('child.stdout','child.stderr'):
            p=directory/name
            if p.exists(): evidence[str(p)]=sha(p,final_check)
        for p,value in {**expected,**evidence}.items(): require(sha(p,final_check)==value,'Final control/evidence bytes changed: '+p)
        complete=failure is None and child is not None and child['identity'] is not None and child['exit_code']==0 and not child['signals']
        if complete: require(output_hashes(outputs,output_root,final_check)==result,'Final output bytes changed')
        def publication_bindings():
            # Encode first, then verify every terminal binding immediately before
            # the exclusive write. Large JSON serialization cannot hide mutation.
            final_check()
            for p,value in {**expected,**evidence}.items():
                require(sha(p,final_check)==value,'Publication control/evidence bytes changed: '+p)
            if complete: require(output_hashes(outputs,output_root,final_check)==result,'Publication output bytes changed')
            require(child is None or child['reaped'] and rt.identity(child['pid']) is None,'Publication native closure changed')
            final_check()
        # All hashes and closure are checked inside the same new amendment envelope.
        publish(directory/'owner_terminal.json',dict(schema='adaptgns_sand_B_preflight_amendment_cpu_terminal_v1',
            status='complete' if complete else 'failed',failure=failure,release_sha256=pin,
            child=None if child is None else public(child),child_native_absent=absent,
            input_sha256=expected,evidence_sha256=evidence,output_sha256=result,
            publication_utc=rt.now().isoformat(),publication_monotonic=rt.mono(),
            outer_timeout_identity=outer,owner_identity=own,scientific_admission=False,
            native_hard_guard=r.get('_native_guard'),native_outer_timing=r.get('_native_outer_timing'),
            numeric_input_files_bound=r.get('_numeric_input_files_count',0),
            output_validation_scope='Exact regular-file tree membership and final SHA256. Output JSON/array semantics are inherited from the frozen worker clean exit and require root/downstream review.',
            root_original_tool_exit_and_timeout_owner_child_native_closure_required=True),final_check,publication_bindings)
        return 0 if complete else 1
    except BaseException as e:
        # Never write after the deadline merely to make failure evidence look
        # complete. The raw original timeout/tool exit remains authoritative.
        try: publish(directory/'publication_failed.json',dict(error=type(e).__name__+': '+str(e),failure=failure,
            child=None if child is None else public(child),all_outputs_preserved=True),final_check)
        except BaseException: pass
        return 1
    finally:
        for s,handler in previous.items(): signal.signal(s,handler)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--execute',action='store_true');p.add_argument('--release',type=Path);p.add_argument('--release-sha256')
    p.add_argument('--hard-deadline-monotonic-ns',type=int);p.add_argument('--outer-term-seconds',type=int)
    p.add_argument('--outer-computed-monotonic-ns',type=int)
    a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(status='description_only',execution_authority=False,operations=list(OPERATIONS),
            new_shared_amendment_seconds=600,original_source_phase_remains_incomplete=True,explicit_human_approval_required=True,owner_has_no_model_or_array_imports=True)));return 0
    require(sys.flags.optimize==0 and a.release is not None and digest(a.release_sha256),'Exact nonoptimized release invocation required')
    path=canonical(a.release);r=read_json(dict(path=str(path),sha256=a.release_sha256));structure(r)
    require(NATIVE_GUARD is not None and a.hard_deadline_monotonic_ns==r['hard_deadline_monotonic_ns'],
            'Previously armed literal native guard required')
    r['_native_guard']=NATIVE_GUARD;r['_outer_term_seconds']=a.outer_term_seconds
    r['_outer_computed_monotonic_ns']=a.outer_computed_monotonic_ns
    rt=Runtime();budget=Budget(r,rt,ENTRY_MONOTONIC)
    return execute(r,path,a.release_sha256,rt,budget)


if __name__=='__main__': raise SystemExit(main())
