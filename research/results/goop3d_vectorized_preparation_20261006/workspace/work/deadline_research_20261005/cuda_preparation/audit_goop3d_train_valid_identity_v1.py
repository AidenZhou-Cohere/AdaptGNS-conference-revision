"""Manifest-only exact-identity check; no arrays, inference, or test access."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = {
    "train": "0f0ce1802e202b9faaf86f0a6ce059c286ed454ceb53273cb926980614b0f864",
    "valid": "f79a20101e3926a9aa9ba91c060f694d673a9f58ef8b565ebf895f7dc2793cef",
}


def main():
    manifests = {}
    for split, expected in PINS.items():
        raw = (HERE / "goop3d_structural_metadata_v1" / f"{split}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Changed {split} manifest")
        manifests[split] = json.loads(raw)
    matches = {}
    for kind in ("positions", "trajectory_content"):
        groups = {}
        for split, manifest in manifests.items():
            for row in manifest["records"]:
                digest = row["positions"]["sha256"] if kind == "positions" else row["trajectory_content_sha256"]
                groups.setdefault(digest, []).append({"split": split, "id": row["id"], "source_index": row["source_index"]})
        matches[kind] = [dict(sha256=k, records=v) for k, v in sorted(groups.items()) if len(v) > 1]
    report = {
        "schema": "goop3d_train_valid_exact_identity_manifest_audit_v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": PINS,
        "record_counts": {s: len(m["records"]) for s, m in manifests.items()},
        "duplicate_groups_within_or_across_splits": matches,
        "no_exact_duplicates": not any(matches.values()),
        "scope": "Verified manifest identities only; numerical arrays were checked by prior conversion and numerical gates. This is not a near-duplicate or shared-physics test.",
        "test_accessed": False,
    }
    destination = HERE / "goop3d_train_valid_identity_v1_report.json"
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"record_counts": report["record_counts"], "no_exact_duplicates": report["no_exact_duplicates"]}))


if __name__ == "__main__":
    main()
