#!/usr/bin/env python3
"""Numerical particle-simulation helpers. Use code/evaluate.py for evaluation."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCHEMA = 'adaptgns_goop3d_graph_support_evaluation_v1'
TRAIN_SCHEMA = 'adaptgns_goop3d_graph_support_cuda_training_v2'
TRAINER_SHA = '71d1ec0ac37ef65e2ccc253df0330ea5f9d2ad9ad82bbcb29a24437df9200dd5'
PROBE_SCHEMA = 'adaptgns_goop3d_vectorized_capacity_checkpoint_v1'
METADATA_SHA = '727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55'
TRAIN_MANIFEST_SHA = '0f0ce1802e202b9faaf86f0a6ce059c286ed454ceb53273cb926980614b0f864'
VALID_MANIFEST_SHA = 'f79a20101e3926a9aa9ba91c060f694d673a9f58ef8b565ebf895f7dc2793cef'
GRAPH_SHA = '8ff3db6ef4ecddcc8c3386bfa840b8dcd76bfb57d3eb550515b0c84520d796a6'
CONVERTER_SHA = '49dece28bf4494591379b8a667e5366b5bdf1609c4dc0757ceb31664c28f9b74'
READER_SHA = '52822d4373097dd0341f47f58fa8121b443160022540edaee0b9fcb831ba043b'
CONTEXT_SHA = '5eb6818ae2699c57e62e80f63724248e8573e4536195eb471df0c647122fb1a5'
CENSUS_SOURCE_SHA = '5364e2042fb9464fe2be5f9f849ea73ff12cc66096bd5da0cbd3e3bff1d7e964'
NUMERICAL_SOURCE_PINS = {
    'research/graph_convention_bridge.py': 'b4a1c2c2ca29a7134f59b94cf85e71c9dcd9235e2c59d59637612f0f62e2d6f7',
    'research/full_rollout.py': '70068dc81aae9190884f2e9888bdec82e9cf80c8f65cef8303011d7fd0c4eee8',
    'research/full_same_state.py': '2efd2cf53394b7214f59b322d256fcb64f2e500b4cfc7ff59ae0cbda7110cfef',
    'research/budget_graph.py': '46df0b5608a4f586a3dbe9653f84c79f6e70e80a2d3bd258e9339406e33a7d59',
}
POLICIES = ('base', 'dense', 'random25', 'speed25', 'laggedrisk25', 'relative-velocity-RMS25')
FRAMES, HORIZON = 301, 295


def require(value, message):
    if not value: raise ValueError(message)




















def grid_indices(count):
    require(type(count) is int and count>0, 'Complete positive source record count required')
    return list(range(count)) if count<=30 else [j*(count-1)//29 for j in range(30)]


def schedules(records, mode, purpose):
    require(all(r['source_index']==i and r['positions']['shape'][0]==FRAMES and r['positions']['shape'][2]==3 for i,r in enumerate(records)), 'Complete orderedT301/D3 records required')
    indices=grid_indices(len(records))
    if purpose=='capacity_timing':
        ranked=sorted(indices,key=lambda i:(records[i]['positions']['shape'][1],i))
        indices=sorted({ranked[0],ranked[(len(ranked)-1)//2],ranked[-1]})
    if mode=='full-rollout':
        return [{'source_index':i,'trajectory_id':records[i]['id'],'particles':records[i]['positions']['shape'][1],
                 'size_group':'fixed_source_grid' if purpose=='final_evaluation' else 'bounded_size_timing'} for i in indices]
    if mode=='clean-validation':
        eligible=len(indices)*HORIZON; flat=[i*(eligible-1)//127 for i in range(128)]
        require(len(set(flat))==128,'128 distinct clean validation histories required')
        pairs=[(indices[i//HORIZON],i%HORIZON+6) for i in flat]
    else:
        targets=(7,80,153,226,300) if purpose=='final_evaluation' else (7,)
        pairs=[(i,t) for i in indices for t in targets]
    return [{'schedule_index':k,'source_index':i,'target_frame':t,'trajectory_id':records[i]['id']} for k,(i,t) in enumerate(pairs)]
















if __name__ == "__main__":
    raise SystemExit("Use code/evaluate.py or code/portable/ entry points.")
