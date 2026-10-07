"""Exact previously reviewed metadata prerequisites; no execution on import."""
from pathlib import Path
from observed_common import read_bound,strict,CAP_JSON

PREP=Path(__file__).resolve().parent.parent
PROXY=PREP/'coder_exact_route_proxy_root_v1.py'
PROXY_SHA='e6bfabe7c2dcf668db83d59480c311cde75d03a3824ff3ec2610ff858d55d57e'
OBSERVER=PREP/'goop3d_post_expiry_metadata_observer_UNADMITTED_transition_v1'
OBSERVER_SHA='e9124cbc677e40ab5fed401c9fefb1da04b06c932e86459d2244ddc75e79eb05'
PROXY_OUTPUT=PREP/'goop3d_observed_fresh_route_root_v1'
OBSERVER_OUTPUT=PREP/'goop3d_observed_fresh_metadata_root_v1'

def check_sources():
    read_bound(PROXY,PROXY_SHA,CAP_JSON,retain=False)
    manifest=strict(read_bound(OBSERVER/'manifest.json',OBSERVER_SHA,CAP_JSON))
    for name,pin in manifest['files_sha256'].items():read_bound(OBSERVER/name,pin,CAP_JSON,retain=False)
    return manifest
