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
SCHEMA = "adaptgns_goop_graph_support_rollout_timing_v1"
ADMISSION_SCHEMA = "adaptgns_goop_rollout_admission_v1"
TRAINING_SCHEMA = "adaptgns_goop_graph_support_cuda_training_v1"
METADATA_SHA = "565d6e13be91be6a6b0fbc31a9aa19ed70a5505228411d0928288fcf874ca3dd"
SOURCES = {
    "train": (1000, 3358535518, "8a0b0cfe3ef56f533abf14d963d5cf632235aa3570cf61a4b42ab655a15cae59"),
    "valid": (30, 93682221, "0e7b261d9ce59491ed79756a1d60bb57324c597a042e257f21b53b52b3c4f79c"),
}
POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25", "relative-velocity-RMS25")
CORE_POLICIES = POLICIES[:-1]
POLICY_SHA = "47df7be9535d40c1d7aadff3ca2b081f44a674fb7bf2c4a35fe1e024b318c665"
TRAINER_SHA = "d761313c8d8cb560de47904a98046dbd21e0c31901127dce279f0a700db91dfd"
DATA_CONTRACT_SHA = "52df979c9e8a74bac9cddc96e479bdafc7c406ec88dfe0f5bbd76dbfbbb4ccb6"
DIAGNOSTIC_SHA = "96a7dc6b6bbbe09c7e09a06947593f1e475eafa4ef6429c7718f205e5ef80960"
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
    spec = importlib.util.spec_from_file_location("_goop_graph_support_policy", policy_path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    return policy.NativeAdapter(native), importlib.import_module("research.full_training")
























if __name__ == "__main__":
    raise SystemExit("Use code/evaluate.py or code/portable/ entry points.")
