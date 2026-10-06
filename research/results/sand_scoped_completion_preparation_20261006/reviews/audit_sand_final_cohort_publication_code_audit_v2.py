"""Synthetic publication-race probes; no actual checkpoints or data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import pytest

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = HERE / "prepare_sand_final_cohort_scoped_v1.py"
    pinned = "ebad4e30edbdb9f444ed07dabf50b31540ecf7f85b134ee059c150d1ff68ca5f"
    assert sha(source) == pinned
    spec = importlib.util.spec_from_file_location("_independent_cohort_fixtures", HERE / "test_prepare_sand_final_cohort_scoped_v1.py")
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    m = test.M
    results = []
    for target in ("process_receipt", "checkpoint", "root_release", "inventory", "copied_root_after_audit"):
        root = Path(tempfile.mkdtemp(prefix="sand_cohort_race_", dir="/private/tmp"))
        with pytest.MonkeyPatch.context() as mp:
            roots, inv, release, sources = test.cohort_fixture.__wrapped__(root, mp)
            rp = test.put(root / "root_release.json", release)
            output = root / "output"
            original_encode = m.encoded
            if target in ("process_receipt", "checkpoint"):
                process = root / "process_A.json"
                cli = ["--execute", "--mode", "inventory", "--host-root", str(roots["A"]), "--process-check", str(process), "--output", str(output)]
                changed = process if target == "process_receipt" else roots["A"] / "jobs/base_seed1/checkpoint-000100000.pt"
                def encode(value):
                    raw = original_encode(value)
                    changed.write_bytes(changed.read_bytes() + b"x")
                    return raw
                mp.setattr(m, "encoded", encode)
            else:
                cli = ["--execute", "--mode", "build", "--inventory-a", str(inv["A"]), "--inventory-b", str(inv["B"]), "--root-release", str(rp), "--output-dir", str(output)]
                for key, path in sources.items():
                    cli += ["--" + key.replace("_", "-"), str(path)]
                if target == "copied_root_after_audit":
                    original_verify = m.verify_bindings
                    calls = []
                    def verify(bindings):
                        original_verify(bindings)
                        calls.append(True)
                        if len(calls) == 2:
                            changed = output / "root_release.json"
                            changed.write_bytes(changed.read_bytes() + b" ")
                    mp.setattr(m, "verify_bindings", verify)
                else:
                    changed = rp if target == "root_release" else inv["A"]
                    def encode(value):
                        raw = original_encode(value)
                        if value.get("schema") == "adaptgns_sand_graph_support_final_cohort_v1":
                            changed.write_bytes(changed.read_bytes() + b" ")
                        return raw
                    mp.setattr(m, "encoded", encode)
            try:
                m.main(cli)
            except (ValueError, FileNotFoundError) as error:
                outcome = "rejected"
                message = str(error)
            else:
                outcome = "accepted"
                message = None
            results.append(dict(target=target, outcome=outcome, error=message, evidence_directory=str(root),
                                final_output_published=(output.exists() if target in ("process_receipt", "checkpoint") else (output / "cohort.json").exists())))
    assert sha(source) == pinned
    report = dict(schema="adaptgns_sand_cohort_publication_independent_probes_v2", source_sha256=pinned,
                  cases=results, scope="Synthetic receipt and opaque checkpoint files only; no actual research data/models")
    path = HERE / "sand_final_cohort_publication_probes_code_audit_v2.json"
    with path.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(path.name, sha(path))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
