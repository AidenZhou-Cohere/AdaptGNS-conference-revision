"""Pure, prospective Goop action-gate mathematics; no data or process access.

The separately frozen protocol controls scientific admission. These helpers
neither load models nor authorize fitting/evaluation from incomplete ledgers.
"""
import json
import numpy as np

PROTOCOL_SHA = "bc9235fae89d32ede8d1e7f846bff07bdcdd85f4671d5922d688a9535f7fe024"
SCHEMA = "adaptgns_goop_global_action_gate_head_v1"
FEATURES = (
    "log1p_particle_count", "speed_mean", "speed_p90", "speed_max",
    "acceleration_difference_mean", "acceleration_difference_p90",
    "acceleration_difference_max", "signed_wall_distance_mean",
    "signed_wall_distance_p10", "outside_particle_fraction",
    "maximum_coordinate_excursion",
)
LAMBDAS = (0.001, 0.1, 10.0)
TRAIN_FRAMES = (6, 62, 118, 174, 231, 287, 343, 400)
VALID_FRAMES = (7, 105, 203, 301, 400)
DOMAINS = {"train": 27101, "valid": 27102, "test": 27103, "train_rollout": 27104}
POLICIES = ("base", "random25", "learned_global_gate", "validation_rate_random_gate")


def require(value, message):
    if not value:
        raise ValueError(message)


def finite_array(value, shape=None):
    value = np.asarray(value, dtype=np.float64)
    require(np.isfinite(value).all(), "Finite float64 array required")
    if shape is not None:
        require(value.shape == shape, "Array shape differs from the fixed contract")
    return value


def features(history):
    """Current six-frame history only; no target, prediction, cache, or RNG."""
    x = finite_array(history)
    require(x.ndim == 3 and x.shape[0] == 6 and x.shape[1] > 0 and x.shape[2] == 2,
            "Goop history must have shape [6,N,2], N>0")
    # This mirrors the coordinate bound; the driver retains the native guard
    # category and input identity before calling this pure helper.
    require(np.max(np.abs(x)) <= 10.0, "History exceeds the scientific coordinate guard")
    speed = np.linalg.norm(x[5] - x[4], axis=1)
    acceleration = np.linalg.norm(x[5] - 2.0 * x[4] + x[3], axis=1)
    wall = np.minimum(x[5] - 0.1, 0.9 - x[5]).min(axis=1)
    out = [np.log1p(x.shape[1]), np.mean(speed), np.quantile(speed, .9, method="linear"),
           np.max(speed), np.mean(acceleration), np.quantile(acceleration, .9, method="linear"),
           np.max(acceleration), np.mean(wall), np.quantile(wall, .1, method="linear"),
           np.mean(wall < 0.0), np.max(np.maximum(-wall, 0.0))]
    return finite_array(out, (len(FEATURES),))


def action_errors(base_prediction, random_prediction, target):
    """Float64 coordinate-MSE label after both actions have been produced."""
    base = finite_array(base_prediction)
    require(base.ndim == 2 and base.shape[0] > 0 and base.shape[1] == 2,
            "Predictions must have shape [N,2], N>0")
    random = finite_array(random_prediction, base.shape)
    truth = finite_array(target, base.shape)
    with np.errstate(over="raise", invalid="raise"):
        base_error = float(np.mean(np.square(base - truth)))
        random_error = float(np.mean(np.square(random - truth)))
    result = dict(base_mse=base_error, random25_mse=random_error,
                  signed_benefit=base_error - random_error)
    require(all(np.isfinite(v) for v in result.values()), "Nonfinite action error")
    return result


def row_ids(split):
    require(split in ("train", "valid"), "Only train and validation label schedules exist")
    count, frames = (1000, TRAIN_FRAMES) if split == "train" else (30, VALID_FRAMES)
    return [(source, frame) for source in range(count) for frame in frames]


def checked_rows(ids, split):
    require(isinstance(ids, (tuple, list)) and len(ids) == len(row_ids(split)),
            "Complete ordered source/frame label ledger required")
    require(all(isinstance(r, (tuple, list)) and len(r) == 2
                and all(type(v) is int for v in r) for r in ids), "Integer source/frame identities required")
    require([tuple(r) for r in ids] == row_ids(split), "Missing, reordered or duplicate label identity")


def rng_material(domain, model_seed, source_index, index, gate=False):
    require(domain in DOMAINS and type(model_seed) is int and model_seed in (0, 1, 2),
            "Fixed RNG domain and paired model seed required")
    count = 1000 if domain in ("train", "train_rollout") else 30
    require(type(source_index) is int and 0 <= source_index < count and type(index) is int,
            "Source and frame/forecast indices required")
    if domain in ("train", "valid"):
        require(index in (TRAIN_FRAMES if domain == "train" else VALID_FRAMES) and not gate,
                "Only scheduled paired-action label RNG is admitted")
    else:
        require(1 <= index <= 395, "Forecast index must be 1..395")
    require(type(gate) is bool, "Gate-domain flag must be boolean")
    return [20261006, DOMAINS[domain], model_seed, source_index, index, 2909 if gate else 1701]


def pair_rng(domain, model_seed, source_index, index):
    return np.random.default_rng(np.random.SeedSequence(rng_material(domain, model_seed, source_index, index)))


def independent_gate(probability, domain, model_seed, source_index, forecast_step):
    require(type(probability) in (int, float) and np.isfinite(probability) and 0 <= probability <= 1,
            "Frozen validation expansion probability required")
    material = rng_material(domain, model_seed, source_index, forecast_step, gate=True)
    return bool(np.random.default_rng(np.random.SeedSequence(material)).random() < probability)


def fit_candidates(x, benefits, ids, model_seed, input_hashes):
    """Fit all three fixed ridge candidates; never select or use validation."""
    checked_rows(ids, "train")
    require(type(model_seed) is int and model_seed in (0, 1, 2), "Seed0/1/2 required")
    require(isinstance(input_hashes, dict) and input_hashes and all(
        isinstance(k, str) and k and isinstance(v, str) and len(v) == 64
        and all(c in "0123456789abcdef" for c in v) for k, v in input_hashes.items()),
        "Named model/data/label input hashes required")
    x = finite_array(x, (8000, len(FEATURES)))
    y = finite_array(benefits, (8000,))
    mu, sigma = np.mean(x, axis=0), np.std(x, axis=0, ddof=0)
    scale = np.maximum(sigma, 1e-12)
    label_mean, label_std = float(np.mean(y)), float(np.std(y, ddof=0))
    label_scale = max(label_std, 1e-12)
    z, t = (x - mu) / scale, (y - label_mean) / label_scale
    gram, rhs = z.T @ z / len(z), z.T @ t / len(z)
    heads = {}
    for penalty in LAMBDAS:
        w = finite_array(np.linalg.solve(gram + penalty * np.eye(len(FEATURES)), rhs), (len(FEATURES),))
        heads[penalty] = dict(schema=SCHEMA, protocol_sha256=PROTOCOL_SHA, model_seed=model_seed,
            feature_names=list(FEATURES), feature_mean=mu.tolist(), feature_std=sigma.tolist(),
            feature_scale=scale.tolist(), label_mean=label_mean, label_std=label_std,
            label_scale=label_scale, coefficients=w.tolist(), ridge_lambda=penalty,
            training_states=8000, training_input_hashes=dict(input_hashes), threshold=0.0,
            label_units="one_step_position_coordinate_mse_difference")
        # Finite training predictions are an admission requirement, not a score
        # for choosing a penalty or deciding whether to retain the experiment.
        predict(heads[penalty], x)
    return heads


def predict(head, x):
    require(head.get("schema") == SCHEMA and head.get("protocol_sha256") == PROTOCOL_SHA
            and head.get("feature_names") == list(FEATURES) and head.get("ridge_lambda") in LAMBDAS
            and head.get("training_states") == 8000 and head.get("threshold") == 0.0,
            "Head contract differs")
    x = finite_array(x)
    require(x.ndim in (1, 2) and x.shape[-1] == len(FEATURES), "Eleven feature columns required")
    mu = finite_array(head["feature_mean"], (len(FEATURES),))
    scale = finite_array(head["feature_scale"], (len(FEATURES),))
    sigma = finite_array(head["feature_std"], (len(FEATURES),))
    w = finite_array(head["coefficients"], (len(FEATURES),))
    require((sigma >= 0).all() and np.array_equal(scale, np.maximum(sigma, 1e-12)),
            "Train population feature scales differ")
    m, s, label_scale = (head[k] for k in ("label_mean", "label_std", "label_scale"))
    require(all(type(v) in (int, float) and np.isfinite(v) for v in (m, s, label_scale))
            and s >= 0 and label_scale == max(s, 1e-12), "Train label normalizers differ")
    with np.errstate(over="raise", invalid="raise"):
        out = finite_array(m + label_scale * (((x - mu) / scale) @ w))
    return float(out) if out.ndim == 0 else out


def select_common_lambda(candidates, validation):
    """One shared penalty; equal frame/source/seed decision loss, larger exact tie."""
    require(isinstance(candidates, dict) and isinstance(validation, dict)
            and all(type(seed) is int for seed in list(candidates) + list(validation))
            and set(candidates) == set(validation) == {0, 1, 2}, "All three integer paired seed identities required")
    scores, diagnostics = {}, {}
    for penalty in LAMBDAS:
        seed_losses, seed_rows = [], {}
        for seed in (0, 1, 2):
            require(set(candidates[seed]) == set(LAMBDAS), "All fixed candidate penalties required")
            head, val = candidates[seed][penalty], validation[seed]
            require(type(head.get("model_seed")) is int and head["model_seed"] == seed
                    and head.get("ridge_lambda") == penalty,
                    "Candidate seed/penalty mapping differs")
            checked_rows(val["row_ids"], "valid")
            x = finite_array(val["features"], (150, len(FEATURES)))
            base = finite_array(val["base_mse"], (150,))
            expanded = finite_array(val["random25_mse"], (150,))
            require((base >= 0).all() and (expanded >= 0).all(), "Coordinate errors cannot be negative")
            predictions = predict(head, x)
            requests = predictions > 0.0
            decision_losses = np.where(requests, expanded, base)
            per_source = decision_losses.reshape(30, 5).mean(axis=1)
            mean = float(per_source.mean())
            seed_losses.append(mean)
            with np.errstate(over="raise", invalid="raise"):
                oracle_regret = float((decision_losses - np.minimum(base, expanded)).reshape(30, 5).mean(axis=1).mean())
                benefit_mse = float(np.square(predictions - (base - expanded)).reshape(30, 5).mean(axis=1).mean())
            require(np.isfinite(per_source).all() and np.isfinite(oracle_regret) and np.isfinite(benefit_mse),
                    "Nonfinite validation diagnostic")
            seed_rows[seed] = dict(decision_mse=mean, per_source_decision_mse=per_source.tolist(),
                validation_requested_expansion_fraction=float(requests.mean()),
                oracle_regret=oracle_regret, signed_benefit_mse=benefit_mse)
        scores[penalty] = float(np.mean(seed_losses))
        diagnostics[penalty] = seed_rows
    require(all(np.isfinite(v) for v in scores.values()), "Nonfinite selection loss")
    selected = min(LAMBDAS, key=lambda penalty: (scores[penalty], -penalty))
    return dict(schema="adaptgns_goop_global_action_gate_selection_v1", protocol_sha256=PROTOCOL_SHA,
        selected_lambda=selected, threshold=0.0, candidate_equal_seed_losses=scores,
        candidate_seed_diagnostics=diagnostics, heads={seed:dict(candidates[seed][selected],
            validation_requested_expansion_fraction=diagnostics[selected][seed]["validation_requested_expansion_fraction"])
            for seed in (0, 1, 2)})


def capacity_label_indices():
    return [i * (8000 - 1) // 63 for i in range(64)]


if __name__ == "__main__":
    print(json.dumps(dict(status="description_only", protocol_sha256=PROTOCOL_SHA,
        features=FEATURES, policies=POLICIES, scientific_execution=False)))
