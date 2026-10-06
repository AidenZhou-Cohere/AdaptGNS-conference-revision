"""Bounded original-architecture WATERDROP training; never evaluates test data.

Default: 100,000 updates, paired seed/step frame and host-noise schedules,
batch two, base graph, original noise, faithful or corrected NLL. Overrides
are labelled explicitly and never relabel a smoke checkpoint as a 100k result.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np
import scipy
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "adaptive-gns"))
from gns import data_loader
from gns.device_utils import resolve_device, runtime_provenance, synchronize
from gns.losses import acceleration_loss
from gns.model_io import build_simulator

HISTORY = 6
NOISE = 6.7e-4
KINEMATIC = 3
SCHEMA = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("Lock PID is invalid; refusing automatic removal")
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Cannot disprove liveness; never clear such a lock.


class RunLock:
    """Exclusive output ownership; stale removal requires an explicit flag.

    A second exclusive recovery lock serializes stale-lock removal. An active
    process, an unreadable PID, or PID reuse is handled conservatively.
    """
    def __init__(self, directory, clear_stale=False):
        self.path = Path(directory)/"run.lock"
        self.clear_stale = clear_stale
        self.payload = {"pid": os.getpid(), "started_utc": utc_now(), "token": uuid.uuid4().hex}
        self.owned = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.clear_stale:
            recovery = self.path.with_name("run.lock.recovery")
            recovery_fd = os.open(recovery, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(recovery_fd, json.dumps(self.payload).encode())
                os.fsync(recovery_fd)
                if self.path.exists():
                    existing = json.loads(self.path.read_text())
                    if pid_alive(existing.get("pid")):
                        raise RuntimeError(f"Run lock belongs to a live process: {existing}")
                    self.path.unlink()
            finally:
                os.close(recovery_fd)
                recovery.unlink()
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise RuntimeError("Output already locked. --clear-stale-lock is allowed only after its PID is verified dead") from error
        try:
            os.write(descriptor, json.dumps(self.payload).encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.owned = True
        return self.payload

    def __exit__(self, *_):
        if self.owned and self.path.exists():
            if json.loads(self.path.read_text()).get("token") == self.payload["token"]:
                self.path.unlink()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def sample_indices(seed, step, n_frames, batch_size=2):
    """Uniform with replacement over every eligible training history.

    Independent per-step RNG; no objective, device or process-global RNG enters.
    Step is the number of completed optimizer updates before this minibatch.
    """
    if min(seed, step) < 0 or n_frames <= 0 or batch_size <= 0:
        raise ValueError("Nonnegative seed/step and positive sizes required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, step, 1103]))
    return rng.integers(n_frames, size=batch_size, dtype=np.int64).tolist()


def host_noise(shape, particle_types, seed, step, noise_std=NOISE):
    """Original twice-integrated random-walk noise, generated only on CPU."""
    n, history, dim = shape
    if history < 2 or n != len(particle_types) or noise_std < 0:
        raise ValueError("Invalid history/noise configuration")
    generator = torch.Generator(device="cpu")
    state = np.random.SeedSequence([seed, step, 2207]).generate_state(1, dtype=np.uint64)
    generator.manual_seed(int(state[0]) % (2**63-1))
    increments = torch.randn((n, history-1, dim), generator=generator, dtype=torch.float32)
    increments *= noise_std/math.sqrt(history-1)
    velocities = increments.cumsum(1)
    noise = torch.cat((torch.zeros((n,1,dim)), velocities.cumsum(1)), 1)
    noise *= (torch.as_tensor(particle_types).cpu() != KINEMATIC).view(-1,1,1)
    return noise


def learning_rate(step, initial=1e-4, final=1e-5, decay_steps=100000):
    if min(initial, final) <= 0 or decay_steps < 1 or step < 0:
        raise ValueError("Invalid learning-rate schedule")
    fraction = min(step/max(decay_steps-1, 1), 1.)
    return initial*(final/initial)**fraction


def validation_schedule(dataset, trajectory_ids, total_frames=128, seed=20261004):
    """Fixed stratified frames; all trajectories represented, no target access."""
    lengths = [int(value) for value in dataset._data_lengths]
    if len(lengths) != len(trajectory_ids) or total_frames < len(lengths):
        raise ValueError("Validation frame budget must represent every trajectory")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3301]))
    counts = np.full(len(lengths), total_frames//len(lengths), dtype=int)
    counts[rng.permutation(len(lengths))[:total_frames % len(lengths)]] += 1
    if any(count > length for count,length in zip(counts,lengths)):
        raise ValueError("A trajectory has too few histories for the fixed validation schedule")
    frames, offset = [], 0
    for trajectory, (length, count) in enumerate(zip(lengths, counts)):
        for local_index in np.sort(rng.choice(length, size=count, replace=False)):
            target_frame = int(local_index)+HISTORY
            frames.append({"dataset_index": offset+int(local_index), "trajectory": trajectory_ids[trajectory],
                           "target_frame": target_frame, "id": f"{trajectory_ids[trajectory]}:{target_frame}"})
        offset += length
    return frames


def load_manifest_dataset(path, expected_split):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "gns-trajectory-manifest" or manifest.get("version") != 1:
        raise ValueError("The full runner requires a version-1 numeric trajectory manifest")
    if manifest.get("split") != expected_split:
        raise ValueError(f"Expected {expected_split} manifest, got {manifest.get('split')}")
    records = manifest["records"]
    ids = [record["id"] for record in records]
    if not records or len(set(ids)) != len(ids):
        raise ValueError("Manifest requires nonempty unique trajectory IDs")
    # The manifest loader verifies SHA256 for every array and metadata file.
    dataset = data_loader.SamplesDataset(str(path), input_length_sequence=HISTORY, verify_hashes=True)
    if len(dataset._data) != len(records) or len(dataset) <= 0:
        raise ValueError("Manifest/sample loader trajectory counts disagree")
    return dataset, manifest, {"manifest_sha256": sha256(path), "source": manifest["source"],
                              "trajectory_ids": ids, "n_trajectories": len(ids), "eligible_frames": len(dataset)}


def frame_identity(dataset, trajectory_ids, index):
    lengths = np.asarray(dataset._data_lengths, dtype=np.int64)
    cumulative = np.cumsum(lengths)
    trajectory = int(np.searchsorted(cumulative, index, side="right"))
    offset = 0 if trajectory == 0 else int(cumulative[trajectory-1])
    return f"{trajectory_ids[trajectory]}:{index-offset+HISTORY}"


def unpack_batch(examples):
    features, labels = data_loader.collate_fn(examples)
    if len(features) != 3:
        raise ValueError("This WATERDROP runner supports positions/types/counts, no material feature")
    position, types, counts = features
    if position.shape[1:] != (HISTORY,2):
        raise ValueError("Expected six-frame 2D WATERDROP histories")
    return position, types, counts, labels


def forward_batch(model, batch, noise, device):
    position, types, counts, labels = batch
    return model(next_positions=labels.to(device), position_sequence_noise=noise.to(device),
                 position_sequence=position.to(device), nparticles_per_example=counts.to(device),
                 particle_types=types.to(device), material_property=None,
                 augment_radius_prob=0., augment_radius_factor=1.267)


@torch.no_grad()
def tensors_are_finite(values):
    values = list(values)
    return bool(values) and bool(torch.isfinite(torch.cat([value.reshape(-1) for value in values])).all())


def assert_finite_gradients(model):
    if not tensors_are_finite(param.grad for param in model.parameters() if param.grad is not None):
        raise FloatingPointError("Nonfinite or missing gradients; optimizer update was not executed")


def capture_rng(device):
    states = {"cpu": torch.get_rng_state()}
    if torch.device(device).type == "mps":
        states["mps"] = torch.mps.get_rng_state()
    return states


def restore_rng(states, device):
    torch.set_rng_state(states["cpu"])
    if torch.device(device).type == "mps":
        torch.mps.set_rng_state(states["mps"])


def save_checkpoint(path, model, optimizer, config, completed_steps, history, device):
    """One atomic file contains a standard format-v2 model plus resume state."""
    synchronize(device)
    payload = {"format_version": 2, "full_training_schema": SCHEMA,
               "state_dict": cpu_tree(model.state_dict()),
               "simulator_config": cpu_tree(getattr(model, "_checkpoint_config", {})),
               "training_config": {"loss": config["objective"], "full_run": config,
                                   "completed_optimizer_updates": completed_steps},
               "optimizer_state": cpu_tree(optimizer.state_dict()), "run_config": config,
               "run_config_sha256": config_hash(config), "completed_steps": int(completed_steps),
               "history": history, "rng_states": cpu_tree(capture_rng(device))}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {"path": path.name, "sha256": sha256(path), "completed_steps": completed_steps}


def restore_checkpoint(path, model, optimizer, expected_config, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("full_training_schema") != SCHEMA:
        raise ValueError("Checkpoint is not an exact-resume full-training checkpoint")
    if payload.get("run_config_sha256") != config_hash(expected_config) or payload["run_config"] != expected_config:
        raise ValueError("Resume configuration/data/source/software hashes differ; start a new explicit run")
    if not 0 <= payload["completed_steps"] <= expected_config["steps"]:
        raise ValueError("Checkpoint completed-step count is inconsistent")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    # torch.optim maps moments to parameter devices on load. Adam's noncapturable
    # step counter intentionally remains on CPU; moving it wholesale changes API semantics.
    restore_rng(payload["rng_states"], device)
    return int(payload["completed_steps"]), payload["history"]


def recover_initialization(output, expected_config):
    """Recover only a run that provably has no completed optimizer updates.

    Called after matching protocol.json and holding the exclusive run lock.
    A complete atomic step-zero checkpoint can recover its missing pointer;
    otherwise initialization is repeated. Any positive-step evidence refuses.
    """
    output = Path(output)
    status_path = output/"status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("completed_steps") != 0 or status.get("run_config_sha256") != config_hash(expected_config):
            raise ValueError("Missing latest pointer with evidence of training or incompatible status; manual inspection required")
    checkpoint = None
    for path in output.glob("checkpoint-*.pt"):
        payload = torch.load(path,map_location="cpu",weights_only=True)
        if (path.name != "checkpoint-000000.pt" or payload.get("completed_steps") != 0
                or payload.get("run_config_sha256") != config_hash(expected_config)):
            raise ValueError("Missing latest pointer with a non-initial checkpoint; refusing restart")
        checkpoint = path
    for path in (output/"validation").glob("*.json"):
        result = json.loads(path.read_text())
        if result.get("completed_steps") != 0 or result.get("run_config_sha256") != config_hash(expected_config):
            raise ValueError("Missing latest pointer with non-initial validation evidence; refusing restart")
    if (output/"history.json").exists():
        saved = json.loads((output/"history.json").read_text())
        if saved.get("training"):
            raise ValueError("Missing latest pointer with training history; refusing restart")
    if checkpoint is None:
        return None
    pointer={"path":checkpoint.name,"sha256":sha256(checkpoint),"completed_steps":0}
    atomic_json(output/"latest.json",pointer)
    return pointer


@torch.no_grad()
def evaluate_validation(model, dataset, schedule, device):
    was_training = model.training
    model.eval()
    records, particle_variance, particle_se, particle_weights = [], [], [], []
    trajectory_counts = {name: sum(row["trajectory"] == name for row in schedule)
                         for name in {row["trajectory"] for row in schedule}}
    try:
        for item in schedule:
            batch = unpack_batch([dataset[item["dataset_index"]]])
            pred, head, target = forward_batch(model, batch, torch.zeros_like(batch[0]), device)
            mask = (batch[1] != KINEMATIC).to(device)
            if not bool(mask.any()):
                raise ValueError("Validation frame has no dynamic particles")
            residual = (pred[mask]-target[mask]).square().sum(-1)
            variance = head[mask].clamp_min(1e-6)
            if not bool(torch.isfinite(residual).all() & torch.isfinite(variance).all()):
                raise FloatingPointError("Nonfinite validation prediction or residual")
            se, q = residual.cpu().double().numpy(), variance.cpu().double().numpy()
            weight = 1/(len(trajectory_counts)*trajectory_counts[item["trajectory"]]*len(se))
            particle_se.append(se); particle_variance.append(q)
            particle_weights.append(np.full(len(se),weight))
            records.append({**item, "dynamic_particles": len(se), "coordinate_mse": float(se.mean()/2),
                            "constant_free_gaussian_nll": float((.5*se/q+np.log(q)).mean()),
                            "realized_vector_se": float(se.mean()), "predicted_vector_se": float(2*q.mean())})
    finally:
        model.train(was_training)
    se, q, weight = map(np.concatenate,(particle_se,particle_variance,particle_weights))
    boundaries = np.unique(np.quantile(q,np.linspace(0,1,11)))
    assignment = np.searchsorted(boundaries[1:-1],q,side="right")
    bins = []
    calibration_gap = 0.
    for index in sorted(set(assignment.tolist())):
        chosen = assignment == index
        mass = float(weight[chosen].sum())
        observed = float(np.sum(weight[chosen]*se[chosen])/mass)
        predicted = float(np.sum(weight[chosen]*2*q[chosen])/mass)
        calibration_gap += mass*abs(observed-predicted)
        bins.append({"index": index, "particles": int(chosen.sum()), "weight": mass,
                     "predicted_vector_se": predicted,"realized_vector_se": observed})
    by_trajectory = {}
    metric_names = ("coordinate_mse","constant_free_gaussian_nll","realized_vector_se","predicted_vector_se")
    for trajectory in sorted(trajectory_counts):
        rows = [row for row in records if row["trajectory"] == trajectory]
        by_trajectory[trajectory] = {name: float(np.mean([row[name] for row in rows])) for name in metric_names}
    return {"history": "clean observed histories; no validation noise; stored noise-adjusted training normalization",
            "aggregation": "equal frames within trajectory, then equal trajectories; dynamic particles within frame",
            "calibration_definition": "weighted absolute bin gap between vector squared residual and 2*variance; quantile boundaries preserve ties",
            "frames": records,"trajectories": by_trajectory,
            "equal_trajectory_mean": {name: float(np.mean([row[name] for row in by_trajectory.values()])) for name in metric_names},
            "binned_vector_se_calibration_gap": calibration_gap,"calibration_bins": bins}


def make_config(args, train_info, valid_info, schedule, device):
    source_paths = [Path(__file__)]+sorted((REPO/"adaptive-gns"/"gns").glob("*.py"))
    sources = {str(path.relative_to(REPO)): sha256(path) for path in source_paths}
    standard = (args.steps == 100000 and args.validation_interval == 5000 and args.validation_frames == 128
                and args.checkpoint_interval == 10000 and args.batch_size == 2
                and args.seed in (0,1,2) and device.type == "mps"
                and train_info["n_trajectories"] == 1000 and valid_info["n_trajectories"] == 30)
    return {"schema": SCHEMA,"scope": "bounded_full_data_100k" if standard else "explicit_smoke_or_protocol_override_not_standard_100k",
            "objective": args.objective,"seed": args.seed,"steps": args.steps,"batch_size": args.batch_size,
            "history": HISTORY,"noise_std": NOISE,"graph": {"radius": .015,"radius_backend": "scipy_host",
            "max_neighbors": 128,"self_loops": True,"training": "base only; no radius augmentation"},
            "architecture": {"width":128,"message_passing_blocks":10,"mlp_layers":2},
            "optimizer": {"kind":"Adam","initial_lr":1e-4,"final_lr":1e-5,"decay_updates":100000,
            "betas":[.9,.999],"eps":1e-8,"weight_decay":0.,"gradient_clipping":None},
            "sampling": "independent SeedSequence(seed,completed_step,stream) per update; uniform over all eligible frames with replacement",
            "validation_interval":args.validation_interval,"validation_frames":schedule,"checkpoint_interval":args.checkpoint_interval,
            "train":train_info,"valid":valid_info,"metadata_sha256":sha256(args.metadata),
            "research_protocol_sha256":sha256(args.protocol),"source_sha256":sources,
            "runtime":{**runtime_provenance(device,"scipy_host"),"platform":platform.platform(),
            "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"threads":args.threads,
            "machine":platform.machine(),"processor":platform.processor(),"logical_cpus":os.cpu_count(),
            "mps_built":torch.backends.mps.is_built(),"mps_available":torch.backends.mps.is_available()},
            "validation_noise":0.,"validation_normalization":"training noise-adjusted scales",
            "checkpoint_selection":"fixed final requested step; never validation/test selection"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest",type=Path,required=True)
    parser.add_argument("--valid-manifest",type=Path,required=True)
    parser.add_argument("--metadata",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--objective",choices=["faithful","nll"],required=True)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--device",choices=["cpu","mps"],default="cpu")
    parser.add_argument("--steps",type=int,default=100000)
    parser.add_argument("--batch-size",type=int,default=2)
    parser.add_argument("--validation-interval",type=int,default=5000)
    parser.add_argument("--validation-frames",type=int,default=128)
    parser.add_argument("--checkpoint-interval",type=int,default=10000)
    parser.add_argument("--log-interval",type=int,default=100)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--protocol",type=Path,default=REPO/"research/protocols/full_waterdrop_100k.md")
    parser.add_argument("--resume",action="store_true",help="Resume only the committed latest checkpoint with exact matching configuration")
    parser.add_argument("--clear-stale-lock",action="store_true",help="Remove an existing run lock only after checking that its PID is dead")
    parser.add_argument("--stop-after",type=int,help="Planned early stop at a safe optimizer boundary; does not change the requested protocol budget")
    args = parser.parse_args()
    if args.seed<0 or min(args.steps,args.batch_size,args.validation_interval,args.validation_frames,args.checkpoint_interval,args.log_interval,args.threads)<=0:
        parser.error("Seed must be nonnegative; all counts/intervals must be positive")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("stop-after must be positive and no greater than requested steps")
    torch.set_num_threads(args.threads)
    device=resolve_device(args.device)
    with RunLock(args.output_dir,args.clear_stale_lock) as process:
        run_training(args,device,process)


def run_training(args,device,process):
    train,train_manifest,train_info=load_manifest_dataset(args.train_manifest,"train")
    valid,valid_manifest,valid_info=load_manifest_dataset(args.valid_manifest,"valid")
    metadata=json.loads(args.metadata.read_text())
    if metadata.get("dim")!=2:
        raise ValueError("This fixed protocol is for 2D WATERDROP")
    metadata_hash=sha256(args.metadata)
    if any(manifest["metadata_sha256"]!=metadata_hash for manifest in (train_manifest,valid_manifest)):
        raise ValueError("Training/validation manifest metadata differs from requested metadata")
    training_hashes={record["trajectory_content_sha256"] for record in train_manifest["records"]}
    if any(record["trajectory_content_sha256"] in training_hashes for record in valid_manifest["records"]):
        raise ValueError("A trajectory is duplicated across train and validation")
    schedule=validation_schedule(valid,valid_info["trajectory_ids"],args.validation_frames)
    config=make_config(args,train_info,valid_info,schedule,device)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    protocol_path=args.output_dir/"protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=config or not args.resume:
            raise ValueError("Existing output requires --resume and an exactly matching protocol")
    elif args.resume:
        raise ValueError("Cannot resume without the original protocol.json")
    else:
        atomic_json(protocol_path,config)
    torch.manual_seed(args.seed)
    model=build_simulator(metadata,NOISE,NOISE,device,connectivity_radius=.015,nmessage_passing_steps=10,
                          uncertainty_parameterization="variance",variance_floor=1e-6,
                          detach_variance_features=args.objective=="faithful",radius_backend="scipy_host").to(device)
    model._training_config={"loss":args.objective,"full_run":config}
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    completed,history=0,{"training":[],"validation":[],"elapsed_seconds":0.}
    latest=None
    if args.resume:
        pointer_path=args.output_dir/"latest.json"
        latest=(json.loads(pointer_path.read_text()) if pointer_path.exists()
                else recover_initialization(args.output_dir,config))
        if latest is not None:
            checkpoint=args.output_dir/latest["path"]
            if sha256(checkpoint)!=latest["sha256"]:
                raise ValueError("Latest checkpoint checksum differs from committed pointer")
            completed,history=restore_checkpoint(checkpoint,model,optimizer,config,device)
    stop=args.steps if args.stop_after is None else args.stop_after
    if stop<completed:
        raise ValueError("Planned stopping step precedes resumed checkpoint")
    started=time.perf_counter()
    prior_elapsed=float(history["elapsed_seconds"])
    def status(state,error=None):
        atomic_json(args.output_dir/"status.json",{"state":state,"completed_steps":completed,
             "pid":process["pid"],"started_utc":process["started_utc"],"updated_utc":utc_now(),
             "requested_steps":args.steps,"scope":config["scope"],"objective":args.objective,"seed":args.seed,
             "run_config_sha256":config_hash(config),"elapsed_seconds":prior_elapsed+time.perf_counter()-started,
             "latest_checkpoint":latest,"last_training":history["training"][-1] if history["training"] else None,
             "last_validation":history["validation"][-1] if history["validation"] else None,"error":error})
    def checkpoint_now():
        nonlocal latest
        history["elapsed_seconds"]=prior_elapsed+time.perf_counter()-started
        path=args.output_dir/f"checkpoint-{completed:06d}.pt"
        latest=save_checkpoint(path,model,optimizer,config,completed,history,device)
        atomic_json(args.output_dir/"latest.json",latest)
    def validate_now():
        result=evaluate_validation(model,valid,schedule,device)
        result.update({"completed_steps":completed,"run_config_sha256":config_hash(config)})
        atomic_json(args.output_dir/"validation"/f"step-{completed:06d}.json",result)
        history["validation"].append({"completed_steps":completed,**result["equal_trajectory_mean"],
                                      "binned_vector_se_calibration_gap":result["binned_vector_se_calibration_gap"]})
    try:
        status("running")
        if completed==0 and not history["validation"]:
            validate_now()
            checkpoint_now()
        while completed<stop:
            step=completed
            indices=sample_indices(args.seed,step,len(train),args.batch_size)
            batch=unpack_batch([train[index] for index in indices])
            noise=host_noise(batch[0].shape,batch[1],args.seed,step)
            rate=learning_rate(step)
            for group in optimizer.param_groups:group["lr"]=rate
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred,head,target=forward_batch(model,batch,noise,device)
            mask=(batch[1]!=KINEMATIC).to(device)
            loss=acceleration_loss(pred,target,mask,pred_variance=head,loss_type=args.objective,variance_floor=1e-6)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss; optimizer update was not executed")
            loss.backward()
            assert_finite_gradients(model)
            optimizer.step()
            if not tensors_are_finite(model.parameters()):
                raise FloatingPointError("Nonfinite parameter after update; last committed checkpoint retained")
            completed+=1
            if completed%args.log_interval==0 or completed in {1,stop}:
                history["training"].append({"completed_steps":completed,"loss":float(loss.detach().cpu()),"lr":rate,
                    "frame_ids":[frame_identity(train,train_info["trajectory_ids"],index) for index in indices],
                    "particles":int(len(batch[0])),"elapsed_seconds":prior_elapsed+time.perf_counter()-started})
                status("running")
                print(json.dumps(history["training"][-1]),flush=True)
            if completed%args.validation_interval==0 or completed==args.steps:
                validate_now()
            if completed%args.checkpoint_interval==0 or completed==stop:
                checkpoint_now()
        status("complete" if completed==args.steps else "planned_stop_incomplete")
        atomic_json(args.output_dir/"history.json",history)
    except BaseException as error:
        status("interrupted" if isinstance(error,KeyboardInterrupt) else "failed",f"{type(error).__name__}: {error}")
        raise


if __name__=="__main__":
    main()
