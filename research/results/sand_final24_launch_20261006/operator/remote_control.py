#!/usr/bin/env python3
"""Inert-until-invoked remote final24 control/observation helper.

No scientific module is imported and no scientific process is launched.
validate_payload is pure; run(payload, runtime) supports synthetic-only tests.
RealRuntime is used solely by the explicit JSON-stdin main entry point.

Common payload keys: action, role, stop_utc, files_sha256, historical_pids,
forbidden_program_tokens, expected_boot_id. verify_stage adds sources (the
23 frozen paths mapped to base64 bytes); stage_release adds release with path,
sha256 and base64; observe_owner adds release_sha256 and expected_owner_argv.
The controller owns phase admission and actual release-schema review. This
helper does not issue a phase/release, launch evaluation, or certify final
stopped closure. All native access is confined to an explicitly invoked runtime.
"""
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import time
import uuid

R = '/root/repos/AdaptGNS-cuda-20261006'
PREP = R + '/cuda_preparation'
CTRL = R + '/sand_final_evaluation_controls_20261006_v1'
PYTHON = R + '/.venv/bin/python'
PYTHON_REAL = '/usr/bin/python3.12'
VENV = R + '/.venv/pyvenv.cfg'
TIMEOUT = '/usr/bin/timeout'
TIMEOUT_SHA = '2db30bc57746c940a0643581cd5e2671e505442bea16164143f0ea8ff4e36453'
ENV = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'LD_LIBRARY_PATH': '/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}
GLOBAL_STOP = '2026-10-07T04:00:00+00:00'
MAX_JSON = 8 * 1024 * 1024
MAX_SOURCE = 2 * 1024 * 1024
BASE_PINS = {'A': {'/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python': '6242e0e8650d7dbdebbc25e08bf4c9359ddaf65f54fcae0e57fa99395fa5357a', '/root/repos/AdaptGNS-cuda-20261006/.venv/pyvenv.cfg': 'b7d360e62970717794ce0ab4f1e1398a49a8a73a138d34ca920b1eb251fe2f3b', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/data_loader.py': '287f26902068d84ea1fd74cccbd37459e415d046a10453f1011c6fecb1330bef', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/device_utils.py': '324db853f98a85a6a2f83755cb1d67e18edac2f503e5dc33d6bd39352749b326', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/graph_network.py': 'af91d949063441ba876fbfbaefceee65f7f16947fe98053720a1ccb15feffe91', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/learned_simulator.py': '216c73a236da6e2441828c4942068cc19618d339859d6492480f07ae2429eadf', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/losses.py': '94ba3f2eb6f103527801f251919675e77e242fff262728f03f0e87cd4e2ea0e5', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/model_io.py': '06599b723814f0be507a3f4210abe10e21f8ada7deeda1acaced07351d82de2e', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py': '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py': '952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/measure_sand_cuda_capacity_v2.py': 'c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md': 'e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_policy.py': '4a1db45f742043ba261c06273c1538fa0195dfb7f203423510b9a353c5ddb03a', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json': 'bc63fabfc07a663ab866f454c07e984dd57b8d4b32aa06a760838c4f84da72e6', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_scoped_operational_amendment_v1.md': '411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_scoped_schedule_fixed_spec_v2.json': '403a96c1304341ea1899c994e0ab7474280add44ea5986ec41570e08d16c337b', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json': 'fbc8ddd94d90adea6e8f4b07caa73931dfcea41e86c8f2f830e3f01f6a177b73', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_goop_evaluation_gpu_scoped_v3.py': 'a13ca1bc30161f89cf79b6c0b632c65a272913302ef0cc37dcfd7ed762c6aa21', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_final_evaluation_scoped_v1.py': 'efebe762ea60b1b711925fb9bb3bc91461a16b1440491b2fa554c99171773829', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_scoped_science_v1.py': '2970922593f41f74713aff9208e4c6964bb1b3f351fee906724bbf52f11d4d13', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py': 'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124', '/root/repos/AdaptGNS-cuda-20261006/research/budget_graph.py': '951f0d13863672f1dd248febf8cc960ba500103ca95361e06c9b0577913a8187', '/root/repos/AdaptGNS-cuda-20261006/research/full_rollout.py': 'b0a37ee47619e699298b86865649e63c402cc1cb5dc2fa3dfd4055f7fa966eb8', '/root/repos/AdaptGNS-cuda-20261006/research/full_same_state.py': 'ff0f9b428791453a1592c23e0c1a4f31653418f74654c859701f52a70f32f592', '/root/repos/AdaptGNS-cuda-20261006/research/full_training.py': 'c6bec98cb8200ef74c3d5d30309dba858c0dbef49f3380a0bc8762683c04e60d', '/root/repos/AdaptGNS-cuda-20261006/research/graph_convention_bridge.py': '2c589c3c762631de5d3b3d60b986cc71178b97b3a76d0ce0d02132247b0be42d', '/root/repos/AdaptGNS-cuda-20261006/research/native_graph_rollout.py': 'b4bbca6660dc449c81958010e30fda8be2bc0aaf58f677e0946d26e600e6daf5', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/metadata.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json': '7c61b7b4633287fe18c74c94fa56f84aa9ce06e9f23002c853eea3a12d8671f4', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json': '0e73e272a4ff386ca1e89bd010c476a82f993ced8f32176144f53b779f32d8ba', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json': '0786ee20abc2333c3bc2b1536f42f5b2f949617a331e70bb7897e9f4ee0e7d37', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/census/all_split_census.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/metadata.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/preflight_test/split_preflight.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/preflight_valid/split_preflight.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/base_seed1/checkpoint-000100000.pt': '68cec3c0e02281c630c5033d51d80450c2ed50e6ec5fbbbc5e17117bbee5d7ec', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/base_seed2/checkpoint-000100000.pt': '84cded9870d2eb51d1ebe0df89c3155d1c88de7cd56ccece95acdd6702bd16a7', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/mix_seed1/checkpoint-000100000.pt': '9b237f5297fd47ff5e214b7ff26dc69a708f023644b2dd94797b69b4c765c6cf', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/mix_seed2/checkpoint-000100000.pt': '15b29c13596d3398f0935703f42d018a520d769dd634581046a1098013865679', '/usr/bin/python3.12': '6242e0e8650d7dbdebbc25e08bf4c9359ddaf65f54fcae0e57fa99395fa5357a'}, 'B': {'/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python': '6242e0e8650d7dbdebbc25e08bf4c9359ddaf65f54fcae0e57fa99395fa5357a', '/root/repos/AdaptGNS-cuda-20261006/.venv/pyvenv.cfg': 'b7d360e62970717794ce0ab4f1e1398a49a8a73a138d34ca920b1eb251fe2f3b', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/data_loader.py': '287f26902068d84ea1fd74cccbd37459e415d046a10453f1011c6fecb1330bef', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/device_utils.py': '324db853f98a85a6a2f83755cb1d67e18edac2f503e5dc33d6bd39352749b326', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/graph_network.py': 'af91d949063441ba876fbfbaefceee65f7f16947fe98053720a1ccb15feffe91', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/learned_simulator.py': '216c73a236da6e2441828c4942068cc19618d339859d6492480f07ae2429eadf', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/losses.py': '94ba3f2eb6f103527801f251919675e77e242fff262728f03f0e87cd4e2ea0e5', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/model_io.py': '06599b723814f0be507a3f4210abe10e21f8ada7deeda1acaced07351d82de2e', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py': '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py': '952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/measure_sand_cuda_capacity_v2.py': 'c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md': 'e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_policy.py': '4a1db45f742043ba261c06273c1538fa0195dfb7f203423510b9a353c5ddb03a', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json': 'bc63fabfc07a663ab866f454c07e984dd57b8d4b32aa06a760838c4f84da72e6', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_scoped_operational_amendment_v1.md': '411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_scoped_schedule_fixed_spec_v2.json': '403a96c1304341ea1899c994e0ab7474280add44ea5986ec41570e08d16c337b', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json': 'fbc8ddd94d90adea6e8f4b07caa73931dfcea41e86c8f2f830e3f01f6a177b73', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_goop_evaluation_gpu_scoped_v3.py': 'a13ca1bc30161f89cf79b6c0b632c65a272913302ef0cc37dcfd7ed762c6aa21', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_final_evaluation_scoped_v1.py': 'efebe762ea60b1b711925fb9bb3bc91461a16b1440491b2fa554c99171773829', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_scoped_science_v1.py': '2970922593f41f74713aff9208e4c6964bb1b3f351fee906724bbf52f11d4d13', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py': 'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124', '/root/repos/AdaptGNS-cuda-20261006/research/budget_graph.py': '951f0d13863672f1dd248febf8cc960ba500103ca95361e06c9b0577913a8187', '/root/repos/AdaptGNS-cuda-20261006/research/full_rollout.py': 'b0a37ee47619e699298b86865649e63c402cc1cb5dc2fa3dfd4055f7fa966eb8', '/root/repos/AdaptGNS-cuda-20261006/research/full_same_state.py': 'ff0f9b428791453a1592c23e0c1a4f31653418f74654c859701f52a70f32f592', '/root/repos/AdaptGNS-cuda-20261006/research/full_training.py': 'c6bec98cb8200ef74c3d5d30309dba858c0dbef49f3380a0bc8762683c04e60d', '/root/repos/AdaptGNS-cuda-20261006/research/graph_convention_bridge.py': '2c589c3c762631de5d3b3d60b986cc71178b97b3a76d0ce0d02132247b0be42d', '/root/repos/AdaptGNS-cuda-20261006/research/native_graph_rollout.py': 'b4bbca6660dc449c81958010e30fda8be2bc0aaf58f677e0946d26e600e6daf5', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/metadata.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json': '7c61b7b4633287fe18c74c94fa56f84aa9ce06e9f23002c853eea3a12d8671f4', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json': '0e73e272a4ff386ca1e89bd010c476a82f993ced8f32176144f53b779f32d8ba', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json': '0786ee20abc2333c3bc2b1536f42f5b2f949617a331e70bb7897e9f4ee0e7d37', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_B_preflight_amendment_20261006_v1/preflight_test/split_preflight.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_B_preflight_amendment_20261006_v1/preflight_valid/split_preflight.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/census/all_split_census.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/metadata.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json': None, '/root/repos/AdaptGNS-cuda-20261006/sand_scoped_science_B_20261006_v1/jobs/base_seed0/checkpoint-000100000.pt': 'f4cbea33853233228e569cc7363851d2f0b396d87fc20f9c80825a7023d8d829', '/root/repos/AdaptGNS-cuda-20261006/sand_scoped_science_B_20261006_v1/jobs/mix_seed0/checkpoint-000100000.pt': 'f7abcf804bacf7d4b21f7557a4823a15c85effd013188548f0b98a0f306beccb', '/usr/bin/python3.12': '6242e0e8650d7dbdebbc25e08bf4c9359ddaf65f54fcae0e57fa99395fa5357a'}}
SOURCES = {'/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/data_loader.py': '287f26902068d84ea1fd74cccbd37459e415d046a10453f1011c6fecb1330bef', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/device_utils.py': '324db853f98a85a6a2f83755cb1d67e18edac2f503e5dc33d6bd39352749b326', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/graph_network.py': 'af91d949063441ba876fbfbaefceee65f7f16947fe98053720a1ccb15feffe91', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/learned_simulator.py': '216c73a236da6e2441828c4942068cc19618d339859d6492480f07ae2429eadf', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/losses.py': '94ba3f2eb6f103527801f251919675e77e242fff262728f03f0e87cd4e2ea0e5', '/root/repos/AdaptGNS-cuda-20261006/adaptive-gns/gns/model_io.py': '06599b723814f0be507a3f4210abe10e21f8ada7deeda1acaced07351d82de2e', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py': '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py': '952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/measure_sand_cuda_capacity_v2.py': 'c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md': 'e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_policy.py': '4a1db45f742043ba261c06273c1538fa0195dfb7f203423510b9a353c5ddb03a', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_scoped_operational_amendment_v1.md': '411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_scoped_schedule_fixed_spec_v2.json': '403a96c1304341ea1899c994e0ab7474280add44ea5986ec41570e08d16c337b', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_goop_evaluation_gpu_scoped_v3.py': 'a13ca1bc30161f89cf79b6c0b632c65a272913302ef0cc37dcfd7ed762c6aa21', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_final_evaluation_scoped_v1.py': 'efebe762ea60b1b711925fb9bb3bc91461a16b1440491b2fa554c99171773829', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_scoped_science_v1.py': '2970922593f41f74713aff9208e4c6964bb1b3f351fee906724bbf52f11d4d13', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py': 'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124', '/root/repos/AdaptGNS-cuda-20261006/research/budget_graph.py': '951f0d13863672f1dd248febf8cc960ba500103ca95361e06c9b0577913a8187', '/root/repos/AdaptGNS-cuda-20261006/research/full_rollout.py': 'b0a37ee47619e699298b86865649e63c402cc1cb5dc2fa3dfd4055f7fa966eb8', '/root/repos/AdaptGNS-cuda-20261006/research/full_same_state.py': 'ff0f9b428791453a1592c23e0c1a4f31653418f74654c859701f52a70f32f592', '/root/repos/AdaptGNS-cuda-20261006/research/full_training.py': 'c6bec98cb8200ef74c3d5d30309dba858c0dbef49f3380a0bc8762683c04e60d', '/root/repos/AdaptGNS-cuda-20261006/research/graph_convention_bridge.py': '2c589c3c762631de5d3b3d60b986cc71178b97b3a76d0ce0d02132247b0be42d', '/root/repos/AdaptGNS-cuda-20261006/research/native_graph_rollout.py': 'b4bbca6660dc449c81958010e30fda8be2bc0aaf58f677e0946d26e600e6daf5'}
OWNER_ARGV = {'A': ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_final_evaluation_scoped_v1.py', '--execute', '--release', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_controls_20261006_v1/A.evaluation_release.json', '--lifecycle-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/measure_sand_cuda_capacity_v2.py', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1'], 'B': ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_final_evaluation_scoped_v1.py', '--execute', '--release', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_controls_20261006_v1/B.evaluation_release.json', '--lifecycle-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/measure_sand_cuda_capacity_v2.py', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_B_20261006_v1']}
CHILD_ARGV = {'A': [['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/base_seed1/checkpoint-000100000.pt', '--checkpoint-sha256', '68cec3c0e02281c630c5033d51d80450c2ed50e6ec5fbbbc5e17117bbee5d7ec', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/base_seed1/full_rollout_test', '--mode', 'full-rollout', '--split', 'test', '--objective', 'faithful', '--arm', 'base', '--seed', '1', '--cuda-index', '0', '--threads', '2', '--max-seconds', '7200'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/base_seed1/checkpoint-000100000.pt', '--checkpoint-sha256', '68cec3c0e02281c630c5033d51d80450c2ed50e6ec5fbbbc5e17117bbee5d7ec', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/base_seed1/same_state_valid', '--mode', 'same-state', '--split', 'valid', '--objective', 'faithful', '--arm', 'base', '--seed', '1', '--cuda-index', '0', '--threads', '2', '--max-seconds', '1800'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/base_seed1/checkpoint-000100000.pt', '--checkpoint-sha256', '68cec3c0e02281c630c5033d51d80450c2ed50e6ec5fbbbc5e17117bbee5d7ec', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/base_seed1/same_state_test', '--mode', 'same-state', '--split', 'test', '--objective', 'faithful', '--arm', 'base', '--seed', '1', '--cuda-index', '0', '--threads', '2', '--max-seconds', '1800'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/base_seed1/checkpoint-000100000.pt', '--checkpoint-sha256', '68cec3c0e02281c630c5033d51d80450c2ed50e6ec5fbbbc5e17117bbee5d7ec', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/base_seed1/clean_validation', '--mode', 'clean-validation', '--split', 'valid', '--objective', 'faithful', '--arm', 'base', '--seed', '1', '--cuda-index', '0', '--threads', '2', '--max-seconds', '900'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/mix_seed1/checkpoint-000100000.pt', '--checkpoint-sha256', '9b237f5297fd47ff5e214b7ff26dc69a708f023644b2dd94797b69b4c765c6cf', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/mix_seed1/full_rollout_test', '--mode', 'full-rollout', '--split', 'test', '--objective', 'faithful', '--arm', 'mix', '--seed', '1', '--cuda-index', '1', '--threads', '2', '--max-seconds', '7200'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/mix_seed1/checkpoint-000100000.pt', '--checkpoint-sha256', '9b237f5297fd47ff5e214b7ff26dc69a708f023644b2dd94797b69b4c765c6cf', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/mix_seed1/same_state_valid', '--mode', 'same-state', '--split', 'valid', '--objective', 'faithful', '--arm', 'mix', '--seed', '1', '--cuda-index', '1', '--threads', '2', '--max-seconds', '1800'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/mix_seed1/checkpoint-000100000.pt', '--checkpoint-sha256', '9b237f5297fd47ff5e214b7ff26dc69a708f023644b2dd94797b69b4c765c6cf', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/mix_seed1/same_state_test', '--mode', 'same-state', '--split', 'test', '--objective', 'faithful', '--arm', 'mix', '--seed', '1', '--cuda-index', '1', '--threads', '2', '--max-seconds', '1800'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/mix_seed1/checkpoint-000100000.pt', '--checkpoint-sha256', '9b237f5297fd47ff5e214b7ff26dc69a708f023644b2dd94797b69b4c765c6cf', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/mix_seed1/clean_validation', '--mode', 'clean-validation', '--split', 'valid', '--objective', 'faithful', '--arm', 'mix', '--seed', '1', '--cuda-index', '1', '--threads', '2', '--max-seconds', '900'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/base_seed2/checkpoint-000100000.pt', '--checkpoint-sha256', '84cded9870d2eb51d1ebe0df89c3155d1c88de7cd56ccece95acdd6702bd16a7', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/base_seed2/full_rollout_test', '--mode', 'full-rollout', '--split', 'test', '--objective', 'faithful', '--arm', 'base', '--seed', '2', '--cuda-index', '2', '--threads', '2', '--max-seconds', '7200'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/base_seed2/checkpoint-000100000.pt', '--checkpoint-sha256', '84cded9870d2eb51d1ebe0df89c3155d1c88de7cd56ccece95acdd6702bd16a7', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/base_seed2/same_state_valid', '--mode', 'same-state', '--split', 'valid', '--objective', 'faithful', '--arm', 'base', '--seed', '2', '--cuda-index', '2', '--threads', '2', '--max-seconds', '1800'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/base_seed2/checkpoint-000100000.pt', '--checkpoint-sha256', '84cded9870d2eb51d1ebe0df89c3155d1c88de7cd56ccece95acdd6702bd16a7', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/base_seed2/same_state_test', '--mode', 'same-state', '--split', 'test', '--objective', 'faithful', '--arm', 'base', '--seed', '2', '--cuda-index', '2', '--threads', '2', '--max-seconds', '1800'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/base_seed2/checkpoint-000100000.pt', '--checkpoint-sha256', '84cded9870d2eb51d1ebe0df89c3155d1c88de7cd56ccece95acdd6702bd16a7', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/base_seed2/clean_validation', '--mode', 'clean-validation', '--split', 'valid', '--objective', 'faithful', '--arm', 'base', '--seed', '2', '--cuda-index', '2', '--threads', '2', '--max-seconds', '900'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/mix_seed2/checkpoint-000100000.pt', '--checkpoint-sha256', '15b29c13596d3398f0935703f42d018a520d769dd634581046a1098013865679', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/mix_seed2/full_rollout_test', '--mode', 'full-rollout', '--split', 'test', '--objective', 'faithful', '--arm', 'mix', '--seed', '2', '--cuda-index', '3', '--threads', '2', '--max-seconds', '7200'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/mix_seed2/checkpoint-000100000.pt', '--checkpoint-sha256', '15b29c13596d3398f0935703f42d018a520d769dd634581046a1098013865679', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/mix_seed2/same_state_valid', '--mode', 'same-state', '--split', 'valid', '--objective', 'faithful', '--arm', 'mix', '--seed', '2', '--cuda-index', '3', '--threads', '2', '--max-seconds', '1800'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/mix_seed2/checkpoint-000100000.pt', '--checkpoint-sha256', '15b29c13596d3398f0935703f42d018a520d769dd634581046a1098013865679', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/mix_seed2/same_state_test', '--mode', 'same-state', '--split', 'test', '--objective', 'faithful', '--arm', 'mix', '--seed', '2', '--cuda-index', '3', '--threads', '2', '--max-seconds', '1800'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/mix_seed2/checkpoint-000100000.pt', '--checkpoint-sha256', '15b29c13596d3398f0935703f42d018a520d769dd634581046a1098013865679', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/mix_seed2/clean_validation', '--mode', 'clean-validation', '--split', 'valid', '--objective', 'faithful', '--arm', 'mix', '--seed', '2', '--cuda-index', '3', '--threads', '2', '--max-seconds', '900']], 'B': [['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_scoped_science_B_20261006_v1/jobs/base_seed0/checkpoint-000100000.pt', '--checkpoint-sha256', 'f4cbea33853233228e569cc7363851d2f0b396d87fc20f9c80825a7023d8d829', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_B_20261006_v1/jobs/base_seed0/full_rollout_test', '--mode', 'full-rollout', '--split', 'test', '--objective', 'faithful', '--arm', 'base', '--seed', '0', '--cuda-index', '2', '--threads', '2', '--max-seconds', '7200'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_scoped_science_B_20261006_v1/jobs/base_seed0/checkpoint-000100000.pt', '--checkpoint-sha256', 'f4cbea33853233228e569cc7363851d2f0b396d87fc20f9c80825a7023d8d829', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_B_20261006_v1/jobs/base_seed0/same_state_valid', '--mode', 'same-state', '--split', 'valid', '--objective', 'faithful', '--arm', 'base', '--seed', '0', '--cuda-index', '2', '--threads', '2', '--max-seconds', '1800'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_scoped_science_B_20261006_v1/jobs/base_seed0/checkpoint-000100000.pt', '--checkpoint-sha256', 'f4cbea33853233228e569cc7363851d2f0b396d87fc20f9c80825a7023d8d829', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_B_20261006_v1/jobs/base_seed0/same_state_test', '--mode', 'same-state', '--split', 'test', '--objective', 'faithful', '--arm', 'base', '--seed', '0', '--cuda-index', '2', '--threads', '2', '--max-seconds', '1800'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_scoped_science_B_20261006_v1/jobs/base_seed0/checkpoint-000100000.pt', '--checkpoint-sha256', 'f4cbea33853233228e569cc7363851d2f0b396d87fc20f9c80825a7023d8d829', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_B_20261006_v1/jobs/base_seed0/clean_validation', '--mode', 'clean-validation', '--split', 'valid', '--objective', 'faithful', '--arm', 'base', '--seed', '0', '--cuda-index', '2', '--threads', '2', '--max-seconds', '900'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_scoped_science_B_20261006_v1/jobs/mix_seed0/checkpoint-000100000.pt', '--checkpoint-sha256', 'f7abcf804bacf7d4b21f7557a4823a15c85effd013188548f0b98a0f306beccb', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_B_20261006_v1/jobs/mix_seed0/full_rollout_test', '--mode', 'full-rollout', '--split', 'test', '--objective', 'faithful', '--arm', 'mix', '--seed', '0', '--cuda-index', '3', '--threads', '2', '--max-seconds', '7200'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_scoped_science_B_20261006_v1/jobs/mix_seed0/checkpoint-000100000.pt', '--checkpoint-sha256', 'f7abcf804bacf7d4b21f7557a4823a15c85effd013188548f0b98a0f306beccb', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_B_20261006_v1/jobs/mix_seed0/same_state_valid', '--mode', 'same-state', '--split', 'valid', '--objective', 'faithful', '--arm', 'mix', '--seed', '0', '--cuda-index', '3', '--threads', '2', '--max-seconds', '1800'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_scoped_science_B_20261006_v1/jobs/mix_seed0/checkpoint-000100000.pt', '--checkpoint-sha256', 'f7abcf804bacf7d4b21f7557a4823a15c85effd013188548f0b98a0f306beccb', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_B_20261006_v1/jobs/mix_seed0/same_state_test', '--mode', 'same-state', '--split', 'test', '--objective', 'faithful', '--arm', 'mix', '--seed', '0', '--cuda-index', '3', '--threads', '2', '--max-seconds', '1800'], ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/valid.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/valid.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_numeric_structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_scoped_science_B_20261006_v1/jobs/mix_seed0/checkpoint-000100000.pt', '--checkpoint-sha256', 'f7abcf804bacf7d4b21f7557a4823a15c85effd013188548f0b98a0f306beccb', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_B_20261006_v1/jobs/mix_seed0/clean_validation', '--mode', 'clean-validation', '--split', 'valid', '--objective', 'faithful', '--arm', 'mix', '--seed', '0', '--cuda-index', '3', '--threads', '2', '--max-seconds', '900']]}
HOSTS = {'A': 'aidenzhou-yellow-worm-77-78fff65d5-62zgv', 'B': 'aidenzhou-aquamarine-toad-75-6d8b45c98d-mgjlq'}
GPU_UUIDS = {'A': ['GPU-b6685200-7eaf-b46c-89b4-53e8715b8e21', 'GPU-38f0a7dd-8a4a-a461-2de3-747ebc7a73c6', 'GPU-af93e07a-ed08-8d07-25c4-4a51969065e0', 'GPU-8e9d199c-b92d-57f0-57af-ffea9caeace7'], 'B': ['GPU-37c3dfdb-9027-e3cb-c899-b0a12f130b52', 'GPU-3b22d55b-8f80-cc77-5a6f-f9181f3286c4', 'GPU-89b83aa8-f20b-dd13-811c-5f54165eb865', 'GPU-28475341-1f76-1575-5aaf-bb8616bc1052']}
PYENV = {'binary_sha256': '6242e0e8650d7dbdebbc25e08bf4c9359ddaf65f54fcae0e57fa99395fa5357a', 'lexical_path': '/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', 'pyvenv_config_path': '/root/repos/AdaptGNS-cuda-20261006/.venv/pyvenv.cfg', 'pyvenv_config_sha256': 'b7d360e62970717794ce0ab4f1e1398a49a8a73a138d34ca920b1eb251fe2f3b', 'resolved_binary_path': '/usr/bin/python3.12', 'sys_base_prefix': '/usr', 'sys_prefix': '/root/repos/AdaptGNS-cuda-20261006/.venv'}

ARRAY_PATHS = frozenset(
    R + prefix + '/' + kind + '_' + f'{i:06d}.npy'
    for prefix in ('/sand_numeric_train_valid_20261006_v1/valid',
                   '/sand_reserved_preparation_20261006_v1/numeric/test')
    for kind in ('position', 'type') for i in range(30))
MANDATORY_FORBIDDEN = frozenset(PREP + '/' + name for name in (
    'supervise_sand_final_evaluation_scoped_v1.py',
    'evaluate_sand_graph_support_final.py',
    'train_sand_graph_support_cuda.py',
    'supervise_sand_scoped_science_v1.py',
    'supervise_sand_B_preflight_amendment_cpu_v1.py',
    'launch_sand_B_preflight_amendment_cpu_v1.py',
    'prepare_sand_reserved_test_recovered_v1.py'))


def need(value, label):
    if not value:
        raise ValueError(label)


def digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def stamp(value):
    need(type(value) is str, 'UTC stop string required')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    need(parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0), 'UTC stop required')
    return parsed


def lexical(path):
    return type(path) is str and path.startswith('/') and str(PurePosixPath(path)) == path and '..' not in PurePosixPath(path).parts


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def decode_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            need(key not in result, 'Duplicate JSON object key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def blob(value, limit):
    need(type(value) is str, 'Base64 text required')
    raw = base64.b64decode(value, validate=True)
    need(len(raw) <= limit, 'Payload file exceeds bounded size')
    return raw


def release_path(role):
    return CTRL + '/' + role + '.evaluation_release.json'


def output_path(role):
    return R + '/sand_final_evaluation_' + role + '_20261006_v1'


def validate_payload(p):
    need(type(p) is dict, 'Object payload required')
    common = {'action', 'role', 'stop_utc', 'files_sha256', 'historical_pids',
              'forbidden_program_tokens', 'expected_boot_id'}
    extra = {'verify_stage': {'sources'}, 'stage_release': {'release'},
             'observe_owner': {'release_sha256', 'expected_owner_argv'}}
    action = p.get('action'); role = p.get('role')
    need(action in extra and role in ('A', 'B'), 'Exact action and role required')
    need(set(p) == common | extra[action], 'Exact control payload keys required')
    end = stamp(p['stop_utc'])
    need(end + timedelta(seconds=11760 + 3600 + 10) <= stamp(GLOBAL_STOP),
         'Full evaluation, analysis and observed-owner clock margins required after control stop')
    need(type(p['expected_boot_id']) is str and str(uuid.UUID(p['expected_boot_id'])) == p['expected_boot_id'], 'Canonical expected boot UUID required')
    pins = p['files_sha256']
    need(type(pins) is dict and set(pins) == set(BASE_PINS[role]) | set(ARRAY_PATHS), 'Exact A164/B162 immutable file key set required')
    need(all(lexical(k) and digest(v) for k, v in pins.items()), 'Nonnull immutable SHA map required')
    need(all(value is None or pins[path] == value for path, value in BASE_PINS[role].items()), 'Frozen candidate pins changed')
    pids = p['historical_pids']
    need(type(pids) is list and all(type(pid) is int and pid > 0 for pid in pids)
         and len(pids) == len(set(pids)), 'Unique positive historical PIDs required')
    tokens = p['forbidden_program_tokens']
    need(type(tokens) is list and len(tokens) == len(set(tokens)) and MANDATORY_FORBIDDEN <= set(tokens)
         and all(lexical(s) and s.startswith(PREP + '/') and s.endswith('.py') for s in tokens), 'Complete canonical forbidden-program tokens required')
    if action == 'verify_stage':
        need(type(p['sources']) is dict and set(p['sources']) == set(SOURCES), 'Exactly 23 frozen source blobs required')
        for path, value in p['sources'].items():
            raw = blob(value, MAX_SOURCE)
            need(hashlib.sha256(raw).hexdigest() == SOURCES[path], 'Frozen source blob pin differs: ' + path)
    elif action == 'stage_release':
        row = p['release']
        need(type(row) is dict and set(row) == {'path', 'sha256', 'base64'} and row['path'] == release_path(role)
             and digest(row['sha256']), 'Exact fresh role release binding required')
        raw = blob(row['base64'], MAX_JSON)
        need(hashlib.sha256(raw).hexdigest() == row['sha256'], 'Release blob digest differs')
        check_release(decode_json(raw), p)
    else:
        need(digest(p['release_sha256']) and p['expected_owner_argv'] == OWNER_ARGV[role], 'Exact observed owner argv and release SHA required')
    return p


def check_release(release, p):
    role = p['role']
    need(type(release) is dict and release.get('schema') == 'adaptgns_sand_evaluation_gpu_scoped_release_v1'
         and release.get('status') == 'admitted_for_execution_allocation' and release.get('issued_by') == 'root'
         and release.get('dataset') == 'Sand' and release.get('host_role') == role and release.get('host') == HOSTS[role], 'Root concrete role release required')
    need(release.get('files_sha256') == p['files_sha256'] and release.get('environment') == ENV
         and release.get('python_environment') == PYENV and release.get('gpu_uuids') == GPU_UUIDS[role], 'Release must bind exact checked runtime and inputs')
    commands = [command for stream in release['streams'] for command in stream['commands']]
    need(commands == CHILD_ARGV[role], 'Frozen scientific child argv changed')
    return release


class Guard:
    def __init__(self, runtime, p):
        self.rt = runtime
        self.end = stamp(p['stop_utc'])
        self.first = runtime.clock()
        self.boot = p['expected_boot_id']
        need(self.first['host_boot_id'] == self.boot, 'Host boot differs from admitted preflight')
        self.mono_end = self.first['host_monotonic_seconds'] + (self.end - stamp(self.first['host_utc'])).total_seconds()
        self.check()

    def check(self):
        sample = self.rt.clock()
        need(sample['host_boot_id'] == self.boot, 'Host boot changed')
        mono = sample['host_monotonic_seconds']; utc = stamp(sample['host_utc'])
        need(type(mono) in (int, float) and math.isfinite(mono), 'Finite native monotonic clock required')
        delta_mono = mono - self.first['host_monotonic_seconds']
        delta_utc = (utc - stamp(self.first['host_utc'])).total_seconds()
        need(delta_mono >= 0 and abs(delta_utc - delta_mono) <= 5, 'Host clock domains changed')
        need(mono + 5 < self.mono_end and utc + timedelta(seconds=5) < self.end, 'Original control stop exhausted')
        return sample


def ordinary(rt, path, kind):
    need(rt.kind(path) == kind and rt.canonical(path), 'Expected canonical ordinary ' + kind + ': ' + path)


def check_runtime(rt, p, guard):
    guard.check()
    info = rt.runtime_identity()
    need(info['hostname'] == HOSTS[p['role']] and info['machine'] == 'aarch64'
         and info['libc'][0] == 'glibc' and bool(info['libc'][1]), 'Exact host/aarch64/glibc runtime required')
    actual_python = info['python_environment']
    need(all(actual_python.get(k) == v for k, v in PYENV.items() if k != 'sys_prefix')
         and actual_python.get('sys_prefix') in ('/usr', R + '/.venv') and info['isolated'] is True
         and info['no_site'] is True and info['dont_write_bytecode'] is True
         and info['optimize'] == 0, 'Exact isolated lexical interpreter/venv required')
    runtime_pins = {PYTHON: PYENV['binary_sha256'], PYTHON_REAL: PYENV['binary_sha256'],
                    VENV: PYENV['pyvenv_config_sha256'], TIMEOUT: TIMEOUT_SHA}
    actual = {path: rt.hash_file(path, guard.check) for path in runtime_pins}
    need(actual == runtime_pins, 'Runtime binary/config hashes changed')
    return actual, info


def check_files(rt, p, guard):
    actual = {}
    for path, pin in p['files_sha256'].items():
        if path == PYTHON:
            need(rt.python_link_valid(), 'Lexical Python symlink changed')
        else:
            ordinary(rt, path, 'file')
        actual[path] = rt.hash_file(path, guard.check)
        need(actual[path] == pin, 'Immutable file hash differs: ' + path)
    return actual


def read_release(rt, p, expected_sha, guard):
    path = release_path(p['role']); ordinary(rt, path, 'file')
    guard.check(); raw = rt.read_bytes(path, MAX_JSON); guard.check()
    need(hashlib.sha256(raw).hexdigest() == expected_sha, 'Staged release SHA changed')
    return check_release(decode_json(raw), p)


def check_native_gpu(rt, p, guard, observing=False):
    guard.check()
    absent = {str(pid): not rt.pid_exists(pid) for pid in p['historical_pids']}
    need(all(absent.values()), 'Historical owned process still exists')
    rows = rt.processes()
    owner = None; children = []
    if observing:
        owners = [row for row in rows if row['argv'] == OWNER_ARGV[p['role']]]
        need(len(owners) == 1 and owners[0]['executable'] == PYTHON_REAL, 'Exactly one native frozen owner required')
        owner = owners[0]
        rt.require_environment(owner['pid'], ENV, ['CUDA_VISIBLE_DEVICES'])
        for row in rows:
            if row['argv'] in CHILD_ARGV[p['role']]:
                need(row['executable'] == PYTHON_REAL and row['ppid'] == owner['pid']
                     and row['pgid'] == row['sid'] == row['pid'], 'Known child native ownership differs')
                rt.require_environment(row['pid'], ENV, ['CUDA_VISIBLE_DEVICES'])
                children.append(row)
        need(len({tuple(row['argv']) for row in children}) == len(children), 'Duplicate active child argv')
    allowed = {row['pid'] for row in children} | ({owner['pid']} if owner else set())
    for row in rows:
        if any(token in row['argv'] for token in p['forbidden_program_tokens']):
            need(row['pid'] in allowed, 'Other matching science/owner process is active')
    guard.check()
    inventory, apps = rt.gpu_snapshot(guard)
    need(inventory == GPU_UUIDS[p['role']], 'Actual all-four index-to-UUID mapping differs')
    owned_indices = set(range(4)) if p['role'] == 'A' else {2, 3}
    child_by_pid = {row['pid']: row for row in children}
    for app in apps:
        need(type(app['pid']) is int and app['pid'] > 0 and app['gpu_uuid'] in inventory, 'Malformed GPU process inventory')
        index = inventory.index(app['gpu_uuid'])
        if app['pid'] in child_by_pid:
            command = child_by_pid[app['pid']]['argv']; expected = int(command[command.index('--cuda-index') + 1])
            need(index == expected and index in owned_indices, 'Owned child on wrong physical GPU')
        else:
            need(index not in owned_indices, 'Foreign compute process on assigned GPU')
    if owner:
        need(rt.identity(owner['pid']) == owner, 'Observed owner identity changed during checks')
        for child in children:
            need(rt.identity(child['pid']) == child, 'Observed child identity changed during checks')
    guard.check()
    return {'native_absent': absent, 'historical_absence': absent,
            'owner': owner, 'children': children, 'gpu_uuids': inventory,
            'gpu_processes': apps, 'other_matching_science_processes': []}


def run(payload, runtime):
    p = validate_payload(payload); rt = runtime; guard = Guard(rt, p)
    rt.require_outer_timeout(guard)
    ordinary(rt, str(PurePosixPath(R).parent), 'directory'); ordinary(rt, R, 'directory')
    runtime_sha, runtime_info = check_runtime(rt, p, guard)
    observed = p['action'] == 'observe_owner'
    if not observed:
        need(rt.kind(output_path(p['role'])) == 'absent', 'Role owner output root must remain absent')
    files_staged = []
    if p['action'] == 'verify_stage':
        need(rt.kind(release_path(p['role'])) == 'absent', 'Actual role release must remain absent')
        if rt.kind(CTRL) == 'absent':
            guard.check(); rt.mkdir(CTRL); files_staged.append({'path': CTRL, 'kind': 'directory'})
        ordinary(rt, CTRL, 'directory')
        for path, pin in SOURCES.items():
            ordinary(rt, str(PurePosixPath(path).parent), 'directory')
            if rt.kind(path) == 'absent':
                raw = blob(p['sources'][path], MAX_SOURCE); guard.check(); rt.write_new(path, raw)
                files_staged.append({'path': path, 'sha256': pin})
            ordinary(rt, path, 'file')
            need(rt.hash_file(path, guard.check) == pin, 'Source differs; overwriting forbidden')
    else:
        ordinary(rt, CTRL, 'directory')
    inputs = check_files(rt, p, guard)
    native = check_native_gpu(rt, p, guard, observed)
    release_sha = None
    if p['action'] == 'stage_release':
        row = p['release']; need(rt.kind(row['path']) == 'absent', 'Release overwrite forbidden')
        guard.check(); rt.write_new(row['path'], blob(row['base64'], MAX_JSON))
        release_sha = row['sha256']; read_release(rt, p, release_sha, guard)
        files_staged.append({'path': row['path'], 'sha256': release_sha})
    elif observed:
        release_sha = p['release_sha256']; read_release(rt, p, release_sha, guard)
    result = {'schema': 'sand_final24_remote_control_observation_v1', 'action': p['action'],
              'role': p['role'], 'hostname': HOSTS[p['role']], 'same_control_stop_utc': p['stop_utc'],
              'runtime_sha256': runtime_sha, 'input_sha256': inputs, 'files_staged': files_staged,
              'runtime': runtime_info, 'required_launch_environment': ENV,
              'required_launch_variables_absent': ['CUDA_VISIBLE_DEVICES'],
              **native, 'release_sha256': release_sha,
              'scientific_execution_performed': False, 'scientific_admission': False,
              'scope': 'Root-bound staging and native observation only; no scientific process launched and no arrays decoded.'}
    encoded(result)  # Complete serialization before the final rechecks and fresh sample.
    need(check_runtime(rt, p, guard) == (runtime_sha, runtime_info), 'Runtime changed before publication')
    need(check_files(rt, p, guard) == inputs, 'Inputs changed before publication')
    final_native = check_native_gpu(rt, p, guard, observed)
    if observed:
        need(final_native['owner'] == native['owner'], 'Owner identity changed before publication')
    result.update(final_native)
    if release_sha:
        read_release(rt, p, release_sha, guard)
    else:
        need(rt.kind(release_path(p['role'])) == 'absent', 'Actual release appeared during staging verification')
    if not observed:
        need(rt.kind(output_path(p['role'])) == 'absent', 'Role output appeared during control action')
    encoded(result)
    result['clock'] = guard.check()
    return result


class RealRuntime:
    def clock(self):
        return {'host_utc': datetime.now(timezone.utc).isoformat(),
                'host_monotonic_seconds': time.monotonic(),
                'host_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}

    def canonical(self, path):
        return lexical(path) and str(Path(path).resolve()) == path

    def kind(self, path):
        p = Path(path)
        if p.is_symlink(): return 'symlink'
        if not p.exists(): return 'absent'
        if p.is_file(): return 'file'
        if p.is_dir(): return 'directory'
        return 'other'

    def python_link_valid(self):
        return Path(PYTHON).is_symlink() and str(Path(PYTHON).resolve()) == PYTHON_REAL

    def hash_file(self, path, check):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while True:
                check(); block = stream.read(1 << 20); check()
                if not block: return h.hexdigest()
                h.update(block)

    def read_bytes(self, path, limit):
        with Path(path).open('rb') as stream: raw = stream.read(limit + 1)
        need(len(raw) <= limit, 'Control file exceeds bounded size')
        return raw

    def mkdir(self, path):
        Path(path).mkdir(mode=0o700, exist_ok=False)

    def write_new(self, path, raw):
        with Path(path).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())

    def runtime_identity(self):
        return {'hostname': platform.node(), 'machine': platform.machine(), 'libc': list(platform.libc_ver()),
                'python_environment': {'lexical_path': sys.executable,
                    'resolved_binary_path': str(Path(sys.executable).resolve()),
                    'binary_sha256': self.hash_file(sys.executable, lambda: None),
                    'sys_prefix': sys.prefix, 'sys_base_prefix': sys.base_prefix,
                    'pyvenv_config_path': VENV, 'pyvenv_config_sha256': self.hash_file(VENV, lambda: None)},
                'isolated': bool(sys.flags.isolated), 'no_site': bool(sys.flags.no_site),
                'dont_write_bytecode': bool(sys.flags.dont_write_bytecode), 'optimize': sys.flags.optimize}

    def pid_exists(self, pid):
        return Path('/proc', str(pid)).exists()

    def identity(self, pid):
        root = Path('/proc', str(pid))
        before = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = [part.decode(errors='surrogateescape') for part in (root / 'cmdline').read_bytes().split(b'\0') if part]
        executable = os.readlink(root / 'exe')
        after = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        need([before[i] for i in (1, 2, 3, 19)] == [after[i] for i in (1, 2, 3, 19)], 'Native identity changed during read')
        return {'pid': pid, 'ppid': int(after[1]), 'pgid': int(after[2]), 'sid': int(after[3]),
                'start_ticks': int(after[19]), 'executable': executable, 'argv': argv}

    def processes(self):
        rows = []
        for path in Path('/proc').iterdir():
            if not path.name.isdigit(): continue
            try:
                row = self.identity(int(path.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
            if row['argv']: rows.append(row)
        return rows

    def require_environment(self, pid, required, absent):
        entries = Path('/proc', str(pid), 'environ').read_bytes().split(b'\0')
        env = {}
        for row in entries:
            if b'=' in row:
                key, value = row.split(b'=', 1)
                key = key.decode(errors='surrogateescape')
                need(key not in env, 'Duplicate native environment key')
                env[key] = value.decode(errors='surrogateescape')
        need(all(env.get(k) == v for k, v in required.items()) and all(k not in env for k in absent), 'Native owner/child environment differs')

    def gpu_snapshot(self, guard):
        def query(fields):
            sample = guard.check(); remaining = min(5, (guard.end - stamp(sample['host_utc'])).total_seconds() - 5)
            need(remaining > 0, 'No GPU query time remains')
            value = subprocess.run(['nvidia-smi', fields, '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=remaining, check=True)
            guard.check()
            return [line.strip().split(',') for line in value.stdout.splitlines() if line.strip()]
        gpu = query('--query-gpu=index,uuid')
        need(len(gpu) == 4 and all(len(row) == 2 for row in gpu), 'Complete GPU inventory required')
        mapping = {int(i.strip()): value.strip() for i, value in gpu}
        need(set(mapping) == set(range(4)) and len(set(mapping.values())) == 4, 'Exact four physical GPU indices required')
        apps = query('--query-compute-apps=pid,gpu_uuid')
        need(all(len(row) == 2 for row in apps), 'Malformed compute inventory')
        return [mapping[i] for i in range(4)], [{'pid': int(pid.strip()), 'gpu_uuid': value.strip()} for pid, value in apps]

    def require_outer_timeout(self, guard):
        need(sys.platform.startswith('linux'), 'Linux only')
        own = self.identity(os.getpid()); parent = self.identity(os.getppid())
        argv = parent['argv']
        need(parent['executable'] == TIMEOUT and len(argv) > 4
             and argv[:3] == [TIMEOUT, '--signal=TERM', '--kill-after=5s']
             and re.fullmatch('[1-9][0-9]*s', argv[3]) is not None
             and argv[4:] == own['argv'] and own['argv'][:5] == [PYTHON, '-I', '-S', '-B', '-c'],
             'Exact bounded remote GNU timeout parent and isolated helper argv required')
        seconds = int(argv[3][:-1]); sample = guard.check()
        need(seconds <= 120 and seconds + 5 < (guard.end - stamp(sample['host_utc'])).total_seconds(),
             'Remote timeout must fit original control stop including kill tail')


def main():
    raw = sys.stdin.buffer.read(MAX_JSON + 1)
    need(len(raw) <= MAX_JSON, 'Control payload exceeds bound')
    result = run(decode_json(raw), RealRuntime())
    sys.stdout.buffer.write(encoded(result)); sys.stdout.buffer.flush()


if __name__ == '__main__':
    main()
