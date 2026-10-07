#!/usr/bin/env python3
"""Numerical particle-simulation helpers. Use code/evaluate.py for evaluation."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import socket
import sys
import time
import traceback

sys.dont_write_bytecode = True
SCHEMA = "adaptgns_sand_graph_support_rollout_timing_v1"
ADMISSION_SCHEMA = "adaptgns_sand_rollout_admission_v1"
TRAINING_SCHEMA = "adaptgns_sand_graph_support_cuda_training_v1"
METADATA_SHA = "cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0"
SOURCES = {
    "train": (1000, 2676940383, "e0b7f68b50702f1af3edfa828b6097e3b28158ddb57631491e60a29658318f7d"),
    "valid": (30, 82712898, "f9c29861107b481ad5eb4a340f4f1016fcaf4050cd6cfc8ae3056f148ab6d903"),
}
POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25", "relative-velocity-RMS25")
CORE_POLICIES = POLICIES[:-1]
POLICY_SHA = "47df7be9535d40c1d7aadff3ca2b081f44a674fb7bf2c4a35fe1e024b318c665"
TRAINER_SHA = "8a43893be22eb59964caebdebe5b864868e2ae8e6ee55fe42ddac5d4b8ab3d15"
DIAGNOSTIC_SHA = "fa1121a73e2a22969403bf269cf02b9446453585c8efb06f0ed836c706753a58"
SOURCE_PINS = {
    "research/native_graph_rollout.py": "788ceb5bba6f9e8c532ba5d2758b3b6a5f4c0e778dc919d4776268d4cf0c3753",
    "research/graph_convention_bridge.py": "b4a1c2c2ca29a7134f59b94cf85e71c9dcd9235e2c59d59637612f0f62e2d6f7",
    "research/full_rollout.py": "70068dc81aae9190884f2e9888bdec82e9cf80c8f65cef8303011d7fd0c4eee8",
    "research/full_same_state.py": "2efd2cf53394b7214f59b322d256fcb64f2e500b4cfc7ff59ae0cbda7110cfef",
    "research/budget_graph.py": "46df0b5608a4f586a3dbe9653f84c79f6e70e80a2d3bd258e9339406e33a7d59",
    "research/full_training.py": "64c925ef198dc2fd032925abdf2f3b5af87d0ce8d29ca9955d21777f899d3a53",
    "adaptive-gns/gns/data_loader.py": "287f26902068d84ea1fd74cccbd37459e415d046a10453f1011c6fecb1330bef",
    "adaptive-gns/gns/device_utils.py": "324db853f98a85a6a2f83755cb1d67e18edac2f503e5dc33d6bd39352749b326",
    "adaptive-gns/gns/graph_network.py": "af91d949063441ba876fbfbaefceee65f7f16947fe98053720a1ccb15feffe91",
    "adaptive-gns/gns/learned_simulator.py": "216c73a236da6e2441828c4942068cc19618d339859d6492480f07ae2429eadf",
    "adaptive-gns/gns/losses.py": "94ba3f2eb6f103527801f251919675e77e242fff262728f03f0e87cd4e2ea0e5",
    "adaptive-gns/gns/model_io.py": "06599b723814f0be507a3f4210abe10e21f8ada7deeda1acaced07351d82de2e",
}
MANIFEST_KEYS = {"format", "version", "split", "dataset", "source", "metadata", "metadata_sha256",
                 "record_count", "records", "converter_sha256"}
SOURCE_KEYS = {"family", "dataset", "file", "size_bytes", "sha256", "ZIP_CRC_verified",
               "member_count", "acquisition_report_sha256"}
RECORD_KEYS = {"id", "source_index", "source_member", "source_inner_index", "positions", "particle_types",
               "trajectory_content_sha256", "logical_content_sha256"}
ARRAY_KEYS = {"path", "shape", "dtype", "size_bytes", "sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)






def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()














def load_helpers(repo):
    repo = Path(repo).resolve()
    require({name: sha(repo / name) for name in SOURCE_PINS} == SOURCE_PINS, "Frozen numerical helper source mismatch")
    names = {relative.removeprefix("adaptive-gns/").removesuffix(".py").replace("/", "."): repo / relative
             for relative in SOURCE_PINS}
    for name, path in names.items():
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve() == path.resolve(), "Unrelated cached module: " + name)
    sys.path[:0] = [str(repo), str(repo / "adaptive-gns")]
    native = importlib.import_module("research.native_graph_rollout")
    for name, path in names.items():
        require(name in sys.modules and Path(sys.modules[name].__file__).resolve() == path.resolve(),
                "Imported helper is not pinned: " + name)
    require(tuple(native.POLICIES) == CORE_POLICIES, "Pinned policy contract differs")
    policy_path = Path(__file__).with_name("sand_graph_support_policy.py")
    require(sha(policy_path) == POLICY_SHA, "Physical policy source differs")
    spec = importlib.util.spec_from_file_location("_sand_graph_support_policy", policy_path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    return policy.NativeAdapter(native), importlib.import_module("research.full_training")
























if __name__ == "__main__":
    raise SystemExit("Use code/evaluate.py or code/portable/ entry points.")
