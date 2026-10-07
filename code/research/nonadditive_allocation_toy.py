"""Particle-simulation methods and data utilities."""

import argparse
import hashlib
import itertools
import json
import math
import platform
import random
from pathlib import Path


def subsets(items, size):
    return [frozenset(s) for s in itertools.combinations(items, size)]


def quadratic_check(seed):
    rng = random.Random(seed)
    n = 6
    budget = 3
    edges = range(n)
    delta = 0.2
    gamma = 0.7
    singles = [rng.uniform(-1, 1) for _ in edges]
    interactions = {
        (i, j): rng.uniform(-gamma, gamma)
        for i, j in itertools.combinations(edges, 2)
    }
    estimated = [a + rng.uniform(-delta, delta) for a in singles]

    def additive(s):
        return sum(singles[i] for i in s)

    def value(s):
        return additive(s) + sum(
            v for (i, j), v in interactions.items() if i in s and j in s
        )

    choices = subsets(edges, budget)
    chosen = max(choices, key=lambda s: sum(estimated[i] for i in s))
    optimum = max(choices, key=value)
    remainders = [value(s) - additive(s) for s in choices]
    epsilon = max(abs(r) for r in remainders)
    regret = value(optimum) - value(chosen)
    overlap_sharp = 2 * len(optimum - chosen) * delta
    bounds = {
        "oscillation": overlap_sharp + max(remainders) - min(remainders),
        "absolute_approximation": 2 * budget * delta + 2 * epsilon,
        "mixed_difference": 2 * budget * delta + budget * (budget - 1) * gamma,
    }
    all_mixed = []
    for e, f in itertools.combinations(edges, 2):
        remaining = [i for i in edges if i not in (e, f)]
        for size in range(budget - 1):
            for context in subsets(remaining, size):
                all_mixed.append(
                    value(context | {e, f}) - value(context | {e})
                    - value(context | {f}) + value(context)
                )
    actual_score_error = max(abs(a - b) for a, b in zip(singles, estimated))
    maximum_mixed = max(abs(v) for v in all_mixed)
    assert actual_score_error <= delta + 1e-12
    assert maximum_mixed <= gamma + 1e-12
    assert epsilon <= math.comb(budget, 2) * gamma + 1e-12
    assert all(regret <= bound + 1e-12 for bound in bounds.values())
    return {
        "seed": seed, "regret": regret, "epsilon_B": epsilon, "bounds": bounds,
        "assumed_score_error_bound": delta, "actual_score_error": actual_score_error,
        "assumed_mixed_difference_bound": gamma,
        "maximum_absolute_mixed_difference": maximum_mixed,
        "enumerated_mixed_differences": len(all_mixed),
    }


def run_check():
    magnitude = 10.0
    candidates = tuple("abcdefgh")

    def synergy(s):
        return float("a" in s) + float("b" in s) + magnitude * ({"c", "d"} <= s)

    budget = int(0.25 * len(candidates))
    choices = subsets(candidates, budget)
    singletons = {e: synergy({e}) for e in candidates}
    chosen = max(choices, key=lambda s: sum(singletons[e] for e in s))
    optimum = max(choices, key=synergy)
    assert chosen == {"a", "b"}
    assert optimum == {"c", "d"}
    assert synergy(optimum) - synergy(chosen) == magnitude - 2

    def triple(s):
        return magnitude * ({"a", "b", "c"} <= s)

    def mixed(e, f, context):
        return (
            triple(context | {e, f})
            - triple(context | {e})
            - triple(context | {f})
            + triple(context)
        )

    baseline_mixed = [mixed(e, f, set()) for e, f in itertools.combinations("abc", 2)]
    assert baseline_mixed == [0.0, 0.0, 0.0]
    assert mixed("b", "c", {"a"}) == magnitude
    return {
        "schema": 1,
        "scope": "synthetic finite set functions only; no model or data access",
        "synergy": {
            "ground_set_size": len(candidates),
            "budget": budget,
            "estimated_singletons": singletons,
            "delta": 0.0,
            "chosen": sorted(chosen),
            "optimum": sorted(optimum),
            "chosen_value": synergy(chosen),
            "optimal_value": synergy(optimum),
            "regret": synergy(optimum) - synergy(chosen),
        },
        "triple_synergy": {
            "baseline_mixed_differences": baseline_mixed,
            "mixed_difference_given_a": mixed("b", "c", {"a"}),
        },
        "quadratic_checks": [quadratic_check(seed) for seed in range(20)],
        "all_checks_passed": True,
        "software": {"python": platform.python_version()},
        "source": {
            "path": "research/nonadditive_allocation_toy.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "reproduction_command": (
            "python -m research.nonadditive_allocation_toy "
            "--output research/results/nonadditive_allocation_toy_check.json"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "results/nonadditive_allocation_toy_check.json",
    )
    args = parser.parse_args()
    result = run_check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output), "all_checks_passed": True,
        "quadratic_checks": len(result["quadratic_checks"]),
        "source_sha256": result["source"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
