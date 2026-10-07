"""Read-only route binding to the separately admitted observed-history phase."""
from pathlib import Path
import datetime as D
import math
import time
from observed_common import need,read_bound,strict,validate_phase,stamp,CAP_JSON

PREP=Path(__file__).resolve().parent.parent
STATE=PREP/'goop3d_observed_history_analysis_released_root_v1'
PHASE=STATE/'analysis_phase.json'
ANCHOR=STATE/'local_phase_anchor.json'

class ObservedPhase:
    def __init__(self,phase_raw,anchor_raw,phase_sha,anchor_sha):
        from observed_common import digest
        need(digest(phase_raw)==phase_sha and digest(anchor_raw)==anchor_sha,'exact phase and original local anchor')
        p,a=strict(phase_raw),strict(anchor_raw);start,stop,_=validate_phase(p)
        need(a['phase_sha256']==phase_sha and a['utc']==start.isoformat(),'one local anchor linked to phase')
        tick=a['monotonic_seconds'];need(type(tick)in(int,float) and math.isfinite(tick) and tick>=0,'finite local anchor')
        self.start=start.timestamp();self.stop=stop.timestamp()-60;self.mono_start=tick;self.mono_stop=tick+3540
        self.metadata={'analysis_phase_path':str(PHASE),'analysis_phase_sha256':phase_sha,'local_phase_anchor_sha256':anchor_sha,'observed_analysis_started_utc':start.isoformat(),'observed_analysis_stop_utc':stop.isoformat(),'route_stop_utc':(stop-D.timedelta(seconds=60)).isoformat(),'route_monotonic_stop':self.mono_stop,'original_phase_modified':False}
    def remaining(self):
        wall=time.time()-self.start;mono=time.monotonic()-self.mono_start
        need(math.isfinite(wall) and math.isfinite(mono) and 0<=wall<3600 and 0<=mono<3600 and abs(wall-mono)<=5,'observed phase expired or UTC/monotonic disagreement')
        remaining=min(self.stop-time.time(),self.mono_stop-time.monotonic())
        need(0<remaining<=3600,'no phase time remains');return remaining
    def expired(self):
        try:self.remaining();return False
        except (ValueError,OverflowError):return True

def load_original_phase(phase_sha,anchor_sha):
    # Compatibility name for the unchanged reviewed proxy serving function.
    phase=ObservedPhase(read_bound(PHASE,phase_sha,CAP_JSON),read_bound(ANCHOR,anchor_sha,CAP_JSON),phase_sha,anchor_sha)
    phase.remaining();return phase
