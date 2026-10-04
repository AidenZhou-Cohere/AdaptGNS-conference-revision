"""Parse complete, CRC-verified records in bounded public TFRecord prefixes.

Writes numeric NPZ arrays only; no pickled object arrays or TensorFlow needed.
Incomplete final records are recorded and excluded, never silently decoded.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import crc32c
import numpy as np
from tfrecord import example_pb2


def masked_crc(data):
    c = crc32c.crc32c(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def parse(path, limit):
    rows, manifest = {}, []
    truncated = False
    with path.open("rb") as f:
        for i in range(limit):
            header = f.read(8)
            if not header:
                break
            if len(header) != 8:
                truncated = True
                break
            crc = f.read(4)
            if len(crc) != 4 or struct.unpack("<I", crc)[0] != masked_crc(header):
                raise ValueError("invalid TFRecord length CRC")
            n = struct.unpack("<Q", header)[0]
            if n > 128 * 1024 * 1024:
                raise ValueError("record exceeds the safety limit")
            raw, crc = f.read(n), f.read(4)
            if len(raw) != n or len(crc) != 4:
                truncated = True
                break
            if struct.unpack("<I", crc)[0] != masked_crc(raw):
                raise ValueError("invalid TFRecord payload CRC")
            ex = example_pb2.SequenceExample()
            ex.ParseFromString(raw)
            pt = np.frombuffer(ex.context.feature["particle_type"].bytes_list.value[0], dtype="<i8")
            pos = np.stack([np.frombuffer(v.bytes_list.value[0], dtype="<f4").reshape(-1, 2)
                            for v in ex.feature_lists.feature_list["position"].feature])
            if pos.shape[1] != len(pt) or not np.isfinite(pos).all():
                raise ValueError("invalid particle positions/types")
            rows[f"position_{i}"] = pos
            rows[f"type_{i}"] = pt
            manifest.append({"index": i, "key": list(ex.context.feature["key"].int64_list.value),
                             "shape": list(pos.shape), "record_sha256": hashlib.sha256(raw).hexdigest()})
    return rows, {"source_file": path.name, "prefix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "records": manifest, "incomplete_final_record": truncated, "requested_limit": limit}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    result = {"sampling": "First complete records from official splits, selected before fitting; pilot convenience sample."}
    seen = set()
    for split, limit in [("train", 8), ("valid", 3), ("test", 3)]:
        rows, record = parse(args.data_dir / f"{split}-prefix.tfrecord", limit)
        if not rows:
            raise ValueError(f"no complete {split} records")
        for item in record["records"]:
            if item["record_sha256"] in seen:
                raise ValueError("duplicate record across splits")
            seen.add(item["record_sha256"])
        np.savez_compressed(args.data_dir / f"{split}-pilot.npz", **rows)
        result[split] = record
        print(split, [r["shape"] for r in record["records"]], flush=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
