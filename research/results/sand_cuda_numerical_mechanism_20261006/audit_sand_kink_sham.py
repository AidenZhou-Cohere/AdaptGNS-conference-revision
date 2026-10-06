"""Read-only numeric artifact audit; no model, CUDA, source data or pickle."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    report_path = args.directory / "report.json"
    assert sha(report_path) == args.report_sha256
    report = json.loads(report_path.read_text())
    assert report["status"] == "diagnostic_complete" and report["all_inputs_reverified"]
    expected = {(case, obj) for case in ("small", "median", "large") for obj in ("faithful", "nll")}
    assert {(r["case"], r["objective"]) for r in report["cases"]} == expected and len(report["cases"]) == 6
    rows, identities = [], {}
    for row in report["cases"]:
        assert row["status"] == "complete" and set(row["phases"]) == set(report["branches"])
        for artifact in list(row["phases"].values()) + [row["relu_localization"]]:
            filename = artifact["artifact_file"]
            path = args.directory / filename
            assert Path(filename).name == filename and sha(path) == artifact["artifact_sha256"]
            identities[filename] = artifact["artifact_sha256"]
        original = args.directory / row["phases"]["cuda_original"]["artifact_file"]
        sham = args.directory / row["phases"]["cuda_native_mask_sham"]["artifact_file"]
        array_count = element_count = 0
        with np.load(original, allow_pickle=False) as a, np.load(sham, allow_pickle=False) as b:
            assert set(a.files) == set(b.files) and len(a.files) == len(set(a.files))
            for key in a.files:
                x, y = a[key], b[key]
                assert not x.dtype.hasobject and x.dtype == y.dtype and x.shape == y.shape
                assert x.tobytes(order="C") == y.tobytes(order="C"), (row["case"], row["objective"], key)
                assert np.isfinite(x).all()
                array_count += 1
                element_count += x.size
        rows.append({"case": row["case"], "objective": row["objective"],
                     "all_sham_artifact_arrays_bitwise_equal": True,
                     "array_count": array_count, "element_count": element_count})
    assert sha(report_path) == args.report_sha256
    print(json.dumps({"schema": "adaptgns_sand_kink_sham_array_audit_v1", "status": "pass",
                      "report_sha256": args.report_sha256, "source_sha256": sha(Path(__file__)),
                      "rows": rows, "all_30_artifact_hashes": identities,
                      "scope": "exact native CUDA sham equivalence for saved outputs/gradients/parameters/updates/Adam tensors; all diagnostic artifact hashes",
                      "no_model_or_cuda_execution": True, "elapsed_seconds": time.perf_counter()-start}, indent=2))


if __name__ == "__main__":
    main()
