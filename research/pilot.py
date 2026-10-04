"""Small CPU WaterDrop experiment; NOT a reproduction of the full paper.

Paired model initializations and minibatch/graph schedules compare corrected
NLL, beta-NLL, and faithful regression. Graph allocation is evaluated on the
same ground-truth states with exactly matched incremental pair budgets.
"""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr
from research.budget_graph import candidates, select_pairs, random_pairs, directed


def mlp(a, b, c):
    return nn.Sequential(nn.Linear(a, b), nn.SiLU(), nn.Linear(b, c))


class PilotGNS(nn.Module):
    def __init__(self, width=48, depth=3):
        super().__init__()
        self.node = mlp(14, width, width)
        self.edge = mlp(3, width, width)
        self.messages = nn.ModuleList([mlp(width * 3, width, width) for _ in range(depth)])
        self.updates = nn.ModuleList([mlp(width * 2, width, width) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.mean = mlp(width, width, 2)
        self.risk = mlp(width, width, 1)

    def forward(self, features, edge_index, edge_features, faithful=False):
        h, e = self.node(features), self.edge(edge_features)
        send, recv = edge_index
        for msg, update, norm in zip(self.messages, self.updates, self.norms):
            e = e + msg(torch.cat((h[send], h[recv], e), dim=-1))
            agg = torch.zeros_like(h).index_add_(0, recv, e)
            h = norm(h + update(torch.cat((h, agg), dim=-1)))
        mean = self.mean(h)
        variance = torch.nn.functional.softplus(self.risk(h.detach() if faithful else h)).squeeze(-1) + 1e-5
        return mean, variance


def objective(mean, variance, target, kind):
    se = (mean - target).square().sum(-1)
    if kind == "faithful":
        return (.5 * se + .5 * se.detach() / variance + target.shape[-1] / 2 * variance.log()).mean()
    nll = .5 * se / variance + target.shape[-1] / 2 * variance.log()
    if kind == "beta_nll":
        nll = variance.detach().sqrt() * nll
    return nll.mean()


def clip_gradients(model, kind):
    # A shared clipping norm would couple risk gradients back into mean updates,
    # defeating faithful regression even though the raw gradients are detached.
    if kind == "faithful":
        torch.nn.utils.clip_grad_norm_([p for n,p in model.named_parameters() if not n.startswith("risk.")], 10.)
        torch.nn.utils.clip_grad_norm_(model.risk.parameters(), 10.)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)


def feature(history, meta):
    velocity = np.diff(history, axis=0)
    velocity = (velocity - np.array(meta["vel_mean"])) / np.array(meta["vel_std"])
    boundary = np.concatenate(((history[-1] - .1) / .015, (.9 - history[-1]) / .015), axis=1)
    return torch.tensor(np.concatenate((velocity.transpose(1, 0, 2).reshape(len(history[-1]), -1),
                                        np.clip(boundary, -1, 1)), axis=1), dtype=torch.float32)


def load_frames(path, meta, count, rng_seed):
    rng = np.random.default_rng(rng_seed)
    z = np.load(path, allow_pickle=False)
    frames = []
    for key in sorted(k for k in z.files if k.startswith("position_")):
        position = z[key]
        particle_type = z[key.replace("position_", "type_")]
        if np.any(particle_type == 3):
            raise ValueError("This simplified pilot supports only dynamic particles")
        steps = np.sort(rng.choice(np.arange(7, len(position)), size=count, replace=False))
        for t in steps:
            history = position[t-6:t]
            target = (position[t] - 2 * position[t-1] + position[t-2] - np.array(meta["acc_mean"])) / np.array(meta["acc_std"])
            frames.append({"id": f"{key}:{t}", "trajectory": key, "step": int(t),
                           "features": feature(history, meta), "prev_features": feature(position[t-7:t-1], meta),
                           "position": history[-1], "prev_position": position[t-2],
                           "speed": np.linalg.norm(history[-1] - history[-2], axis=1),
                           "target": torch.tensor(target, dtype=torch.float32),
                           "graph": candidates(history[-1], .015),
                           "prev_graph": candidates(position[t-2], .015)})
    return frames


def graph_tensors(position, pairs):
    edge = directed(pairs)
    delta = (position[edge[0]] - position[edge[1]]) / .015
    attr = np.concatenate((delta, np.linalg.norm(delta, axis=1, keepdims=True)), axis=1)
    return torch.from_numpy(edge), torch.tensor(attr, dtype=torch.float32)


def run_forward(model, f, pairs, faithful, previous=False):
    ei, ef = graph_tensors(f["prev_position"] if previous else f["position"], pairs)
    return model(f["prev_features"] if previous else f["features"], ei, ef, faithful)


@torch.no_grad()
def evaluate(model, frames, kind, seed):
    model.eval()
    records = []
    rng = np.random.default_rng(91300 + seed)
    for frame in frames:
        graph = frame["graph"]
        budget = math.floor(.25 * len(graph.extra))
        start = time.perf_counter()
        base_mean, risk = run_forward(model, frame, graph.base, kind == "faithful")
        score_seconds = time.perf_counter() - start
        _, prev_risk = run_forward(model, frame, frame["prev_graph"].base, kind == "faithful", previous=True)
        scores = risk.numpy()
        squared = (base_mean - frame["target"]).square().sum(-1).numpy()
        corr = spearmanr(scores, squared).statistic
        strategies = {
            "base": graph.base,
            "dense": np.concatenate((graph.base, graph.extra)),
            "random25": random_pairs(graph, budget, rng),
            "speed25": select_pairs(graph, frame["speed"], budget),
            "current_risk25": select_pairs(graph, scores, budget),
            "lagged_base_risk25": select_pairs(graph, prev_risk.numpy(), budget),
        }
        measurements = {}
        for policy, pairs in strategies.items():
            start = time.perf_counter()
            mean, variance = run_forward(model, frame, pairs, kind == "faithful")
            elapsed = time.perf_counter() - start
            measurements[policy] = {"mse_normalized_acceleration": float((mean-frame["target"]).square().mean()),
                                    "directed_edges": 2 * len(pairs), "forward_seconds": elapsed}
        records.append({"id": frame["id"], "trajectory": frame["trajectory"], "n_particles": len(scores),
                        "extra_pair_budget": budget, "risk_error_spearman": float(corr) if np.isfinite(corr) else None,
                        "realized_vector_se": float(squared.mean()), "predicted_vector_se": float(2*scores.mean()),
                        "current_risk_scoring_seconds": score_seconds, "policies": measurements})
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--objectives", nargs="+", default=["faithful", "nll", "beta_nll"], choices=["faithful", "nll", "beta_nll"])
    args = p.parse_args()
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.data_dir / "metadata.json").read_text())
    train = load_frames(args.data_dir / "train-pilot.npz", meta, 24, 810)
    valid = load_frames(args.data_dir / "valid-pilot.npz", meta, 12, 811)
    test = load_frames(args.data_dir / "test-pilot.npz", meta, 12, 812)
    protocol = {"scope": "Small WaterDrop CPU pilot; not the original architecture, full training, or a benchmark claim.",
                "steps": args.steps, "seeds": args.seeds, "objectives": args.objectives,
                "architecture": {"latent_width": 48, "processor_depth": 3, "history": 6, "features": "normalized velocities and box distances", "noise": 0},
                "training": {"lr": .001, "batch_size": 1, "graph": "random pair fraction uniformly sampled from [0,.25,.5,1] per update", "checkpoint": "fixed final step; no test selection"},
                "budget": "25% of available annulus pairs per identical state, floored; base edges always retained",
                "lag_caveat": "lagged_base_risk uses previous ground-truth state's base graph, not a free-rollout adaptive graph",
                "timing_caveat": "CPU forward timer includes tensor/edge-feature preparation, excludes candidate search and risk prepass; no end-to-end speed claim",
                "frames": {"train": [f['id'] for f in train], "valid": [f['id'] for f in valid], "test": [f['id'] for f in test]},
                "hardware": platform.platform(), "torch": torch.__version__, "threads": torch.get_num_threads(),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print({k: len(v) for k, v in [("train",train),("valid",valid),("test",test)]}, flush=True)
    for seed in args.seeds:
        for kind in args.objectives:
            torch.manual_seed(seed)
            rng = np.random.default_rng(5000 + seed)
            model = PilotGNS()
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            start = time.perf_counter()
            curve = []
            for step in range(args.steps):
                f = train[rng.integers(len(train))]
                fraction = rng.choice([0., .25, .5, 1.])
                pairs = random_pairs(f["graph"], math.floor(fraction * len(f["graph"].extra)), rng)
                mean, variance = run_forward(model, f, pairs, kind == "faithful")
                loss = objective(mean, variance, f["target"], kind)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                optimizer.zero_grad()
                loss.backward()
                clip_gradients(model, kind)
                optimizer.step()
                if (step+1) % 100 == 0:
                    curve.append({"step": step+1, "loss": float(loss.detach()), "seconds": time.perf_counter()-start})
                    print(seed, kind, curve[-1], flush=True)
            seconds = time.perf_counter() - start
            out = {"seed": seed, "objective": kind, "training_seconds": seconds, "curve": curve,
                   "validation": evaluate(model, valid, kind, seed), "test": evaluate(model, test, kind, seed)}
            stem = args.output_dir / f"{kind}_seed{seed}"
            stem.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
            torch.save({"state_dict": model.state_dict(), "protocol": protocol, "seed": seed, "objective": kind}, stem.with_suffix(".pt"))
            print("SAVED", stem.name, seconds, flush=True)


if __name__ == "__main__":
    main()
