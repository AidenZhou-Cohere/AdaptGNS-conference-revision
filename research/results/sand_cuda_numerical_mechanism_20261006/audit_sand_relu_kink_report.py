"""Independent standard-library audit of saved reports; no model/data execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(comparison):
    tensors = list(comparison["tensors"].values())
    assert comparison["names_equal"] and comparison["tensor_count"] == len(tensors)
    assert all(item["finite"] for item in tensors)
    return {
        "passed": comparison["passed"], "tensor_count": len(tensors),
        "outside_tolerance_elements": sum(item["outside_tolerance_elements"] for item in tensors),
        "sign_disagreements": sum(item.get("sign_disagreements", 0) for item in tensors),
        "maximum_tolerance_ratio": max(item["maximum_tolerance_ratio"] for item in tensors),
        "maximum_absolute_error": max(item["max_absolute_error"] for item in tensors),
        "all_reported_bitwise_equal": all(item["bitwise_equal"] for item in tensors),
    }


def main():
    diagnostic_path = ROOT / "sand_relu_kink_v1_report.json"
    validation_path = ROOT / "sand_actual_data_validation_v2_report.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    validation = json.loads(validation_path.read_text())
    assert sha(diagnostic_path) == "d3b9c0a1c6953dc26ba3a00f5a2425231cf5c131984a1b404364d73b7bb9bbe3"
    assert diagnostic["validation_report_sha256"] == sha(validation_path)
    assert diagnostic["source_sha256"] == sha(ROOT / "diagnose_sand_relu_kinks.py")
    assert diagnostic["trainer_sha256"] == validation["trainer_sha256"] == sha(ROOT / "train_sand_cuda_deterministic.py")
    assert validation["source_sha256"] == sha(ROOT / "validate_sand_cuda.py")
    assert diagnostic["status"] == "diagnostic_complete" and diagnostic["all_inputs_reverified"]
    assert diagnostic["prior_validation_status"] == validation["status"] == "validation_failed"
    assert diagnostic["scientific_training_admitted"] is False and validation["scientific_training"] is False
    assert diagnostic["initialization_seed"] == validation["model_initialization_seed"] == 1729
    assert diagnostic["runtime"] == validation["runtime"] and diagnostic["tolerances"] == validation["tolerances"]
    expected = {(size, objective) for size in ("small", "median", "large") for objective in ("faithful", "nll")}
    assert len(diagnostic["cases"]) == 6 and {(row["case"], row["objective"]) for row in diagnostic["cases"]} == expected
    prior = {case["case"]: case for case in validation["cases"]}
    rows = []
    for case in diagnostic["cases"]:
        size, objective = case["case"], case["objective"]
        assert case["status"] == "complete" and case["phase"] == "complete"
        assert case["batch_sha256"] == prior[size]["batch_sha256"]
        assert set(case["phases"]) == set(diagnostic["branches"])
        assert all(phase["status"] == "complete" and phase["saved_tensor_count"] == len(phase["array_paths"])
                   for phase in case["phases"].values())
        source = prior[size]["objectives"][objective]
        parity = source["parity"]
        assert source["encoder_features"]["passed"] and parity["actual_batch_ordered_edges"]["bitwise_equal"]
        assert aggregate(parity["normalization"])["all_reported_bitwise_equal"]
        assert all(value["passed"] for value in parity["outputs"].values())
        assert parity["faithful_gradient_control"]["passed"]
        comparisons = case["comparisons"]
        sham = comparisons["cuda_native_mask_sham"]
        assert aggregate(sham["gradients_vs_cuda_original"])["all_reported_bitwise_equal"]
        assert aggregate(sham["preupdate_outputs_vs_cuda_original_exact"])["all_reported_bitwise_equal"]
        control = comparisons["cuda_cpu_mask_control"]
        assert aggregate(control["gradients_vs_cpu_original"])["passed"]
        assert aggregate(control["gradients_vs_cpu_original"])["sign_disagreements"] == 0
        assert aggregate(control["preupdate_outputs_vs_cuda_original_exact"])["all_reported_bitwise_equal"]
        assert all(value["passed"] for value in control["postupdate_outputs_vs_cpu"].values())
        layers = case["relu_localization"]["layers"]
        assert len(layers) == case["relu_localization"]["module_calls"] == 48
        assert sum(layer["disagreements"] for layer in layers) == case["relu_localization"]["sign_disagreements"]
        row = {"case": size, "objective": objective, "batch_sha256": case["batch_sha256"],
               "relu_sign_disagreements": case["relu_localization"]["sign_disagreements"],
               "maximum_flipped_absolute_preactivation": max(max(layer["disagreement_max_abs_cpu"] or 0,
                   layer["disagreement_max_abs_cuda"] or 0) for layer in layers),
               "sham_reported_bitwise_exact": True, "source_graph_feature_loss_checks_pass": True}
        for branch in ("cuda_original", "cuda_cpu_mask_control"):
            values = comparisons[branch]
            row[branch] = {key: aggregate(values[key]) for key in ("gradients_vs_cpu_original",
                "first_adam_update_vs_cpu", "first_adam_state_vs_cpu")}
            row[branch]["postupdate_outputs_vs_cpu"] = {key: value["passed"] for key, value in values["postupdate_outputs_vs_cpu"].items()}
        rows.append(row)
    assert all(record["passed"] and record["restore_model_optimizer_exact"] and record["restore_rng_exact"]
               and record["resumed_rng_exact"] for record in validation["replay"].values())
    report = {
        "schema": "sand_relu_kink_independent_report_review_v1", "status": "report_review_complete",
        "review_scope": "Saved JSON comparisons/source hashes only; raw tensor artifact verification is separately owned by root",
        "audit_source_sha256": sha(__file__), "diagnostic_report_sha256": sha(diagnostic_path),
        "validation_report_sha256": sha(validation_path), "cases": rows,
        "source_loss_graph_defect_evidence_found": False,
        "interpretation": "Near-zero ReLU-mask differences explain all 13 prior gradient tolerance failures and the dominant sign-flipping first-Adam differences in these six cases. This is a conditional diagnostic intervention, not a corrected production derivative.",
        "remaining_difference": "Small NLL has no ReLU sign disagreements, yet two first-Adam update/state entries remain outside replay tolerance (max absolute difference 6.113201379776001e-6) under the unchanged CPU-mask control. Ordinary rounding and Adam sensitivity remain; the intervention does not establish CPU/CUDA optimizer equivalence.",
        "timing_only_recommendation": "A separately declared native deterministic CUDA feasibility/timing study is supported after root's raw-artifact audit. Use the ordinary CUDA model and optimizer, no mask intervention, and preserve failed CPU/CUDA gates.",
        "strict_cpu_cuda_validation_passed": False, "scientific_training_admitted": False,
        "limitations": "One initialization and three saved actual-data batches do not prove all future training states or long-horizon equivalence. A scientific native-CUDA cohort requires a separately frozen protocol and complete feasibility forecast.",
    }
    path = ROOT / "sand_relu_kink_independent_code_review.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"review_file": str(path), "sha256": sha(path), "cases": len(rows), "status": report["status"]}))


if __name__ == "__main__":
    main()
