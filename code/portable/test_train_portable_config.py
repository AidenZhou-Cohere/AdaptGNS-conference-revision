"""Source/config-only tests. No Torch import, dataset arrays or model execution."""
import copy
import hashlib
import unittest
from train_portable import parse_args, validate_manifest_header


def fixture(dataset):
    frames = 401 if dataset == "goop" else 320
    metadata = {"dim": 2, "default_connectivity_radius": .015, "dt": .0025}
    records = []
    for index in range(1000):
        a, b = hashlib.sha256(f"p{index}".encode()).hexdigest(), hashlib.sha256(f"t{index}".encode()).hexdigest()
        records.append({"id": f"train:{index:06d}", "source_index": index,
                        "positions": {"shape": [frames, 10, 2], "dtype": "<f4", "sha256": a},
                        "particle_types": {"shape": [10], "dtype": "<i8", "sha256": b},
                        "trajectory_content_sha256": hashlib.sha256((a + ":" + b).encode()).hexdigest()})
    return {"format": "gns-trajectory-manifest", "version": 1, "dataset": "Goop" if dataset == "goop" else "Sand",
            "split": "train", "metadata": metadata, "metadata_sha256": "metadata-pin", "record_count": 1000,
            "records": records}, metadata


class PortableConfigTests(unittest.TestCase):
    def test_both_fixed_dataset_contracts(self):
        for name in ("goop", "sand"):
            manifest, metadata = fixture(name)
            self.assertEqual(validate_manifest_header(manifest, metadata, name, "metadata-pin", "metadata-pin")["frames"],
                             401 if name == "goop" else 320)

    def test_no_subset(self):
        manifest, metadata = fixture("goop")
        manifest["records"].pop()
        with self.assertRaises(ValueError):
            validate_manifest_header(manifest, metadata, "goop", "metadata-pin", "metadata-pin")

    def test_no_split_substitution(self):
        manifest, metadata = fixture("sand")
        manifest["split"] = "test"
        with self.assertRaises(ValueError):
            validate_manifest_header(manifest, metadata, "sand", "metadata-pin", "metadata-pin")

    def test_exact_source_order(self):
        manifest, metadata = fixture("sand")
        manifest["records"][0], manifest["records"][1] = manifest["records"][1], manifest["records"][0]
        with self.assertRaises(ValueError):
            validate_manifest_header(manifest, metadata, "sand", "metadata-pin", "metadata-pin")

    def test_no_dtype_change(self):
        manifest, metadata = fixture("goop")
        manifest["records"][0]["positions"]["dtype"] = "<f8"
        with self.assertRaises(ValueError):
            validate_manifest_header(manifest, metadata, "goop", "metadata-pin", "metadata-pin")

    def test_metadata_pin(self):
        manifest, metadata = fixture("goop")
        with self.assertRaises(ValueError):
            validate_manifest_header(manifest, metadata, "goop", "different-pin", "metadata-pin")

    def test_goop_cannot_add_context_input(self):
        manifest, metadata = fixture("goop")
        metadata["context_mean"] = [0]
        with self.assertRaises(ValueError):
            validate_manifest_header(manifest, metadata, "goop", "metadata-pin", "metadata-pin")

    def test_copying_contents_does_not_create_new_trajectory(self):
        manifest, metadata = fixture("sand")
        for field in ("positions", "particle_types", "trajectory_content_sha256"):
            manifest["records"][1][field] = copy.deepcopy(manifest["records"][0][field])
        with self.assertRaises(ValueError):
            validate_manifest_header(manifest, metadata, "sand", "metadata-pin", "metadata-pin")

    def test_fixed_recipe_cli(self):
        args = parse_args(["--dataset", "sand", "--frozen-dir", ".", "--check-source-equivalence"])
        self.assertEqual((args.updates, args.threads, args.objective, args.checkpoint_every, args.log_every, args.stop_after),
                         (100000, 2, "faithful", 10000, 100, None))


if __name__ == "__main__":
    unittest.main()
