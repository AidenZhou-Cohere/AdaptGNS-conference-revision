# Goop-3D train/validation acquisition and conversion

Prepared after the human resume, with Goop2D retaining first priority. Root owns current process/disk checks and execution. This preparation performs no acquisition, official-array read, model/GPU work or test-source access. It leaves the frozen Goop2D and WaterDrop sources unchanged.

Root's bounded first-record inspection at 2026-10-06 04:31:37 UTC succeeded under the original first-record freeze: source-order training record0 has N9271, T301, D3 and stored-trajectory H295 after six initial frames. Both TFRecord CRCs and exact range/generation checks passed. This verifies one record only. It does not establish the full record count, maximum particle count, full-object integrity, auxiliary semantics or training cost. No additional resume release was created before that inspection; the original freeze and unchanged input/source checks are the evidence.

## Exact source objects

Source base: `https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/Goop-3D/`.

| Object | Bytes | GCS generation | Publisher CRC32C |
| --- | ---: | --- | --- |
| metadata.json | 471 | 1599153912104523 | odsy4Q== |
| train.tfrecord | 27448425232 | 1599159646798459 | gBeM1w== |
| valid.tfrecord | 2857385340 | 1599154482348408 | rw9odw== |

Metadata SHA256: `727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55`. The constants were compared by AST with the preserved inventory SHA `32c94925f1306580a5a2e63487da7274e1c44ff480d28819e44a07eabdee67a3`. Metadata fixes dim3, sequence_length300, radius0.025, dt0.0025 and bounds[0.2,0.8] on all three axes.

Whole train+valid+metadata acquisition is 30,305,811,043 bytes. Retaining source and numeric arrays together needs approximately61GB before working-space overhead. Plan at least65GiB free on the chosen destination before starting; this is a conservative operational allowance, not a measured final size. Retained failed attempts and later copies need additional space. Do not extrapolate dataset counts or GPU throughput from one record.

## Prepared source and checks

- `acquire_goop3d_train_valid.py`, SHA `f6f2aa1792de45c99536e185482caee9550b648a0eed2d3b21ec9a24bff9ddb5`.
- `prepare_goop3d_official.py`, SHA `49dece28bf4494591379b8a667e5366b5bdf1609c4dc0757ceb31664c28f9b74`.
- Unchanged generic TFRecord reader `research/prepare_full_waterdrop.py`, SHA `ab077f11240a7296a2a15679ae8a57032c91280fdc94c23b63155e86793d2f33`.

The acquisition adapter is separate from the reviewed Goop2D source. It fetches only the three exact generation-pinned objects, requires status200 without redirects and identity encoding, compares response length/generation/headerCRC, then computes complete stream length/SHA256/CRC32C. It saves full response headers before validation, preserves failed partial files, periodically saves received-byte progress and catches SIGTERM/SIGINT for a failure receipt. There is no automatic retry or resume. A hard kill or host loss may leave a last-progress or temporary receipt; partial files remain evidence and must be inspected before any new attempt.

The conversion adapter reuses the unchanged reader's generic decoder at dimension3, never its WaterDrop-specific converter. It independently hashes the complete source against the completed acquisition receipt, verifies both CRCs of every record through EOF, preserves source order/keys and compares all saved position/type/auxiliary element bytes to the protobuf arrays. It rejects 2D payloads rather than reinterpreting them. Original float32 positions, int64 types, negative zero and auxiliary NaN payload bits remain unchanged. Auxiliaries are marked semantically unreviewed. Exact position/type duplicates within or across the selected splits fail with staging retained. No test source is accepted by the CLI.

Both selected splits must finish before manifest publication. Successful status is `complete_structural_only`, with `scientific_training_admission: false`. The structural report includes complete record counts, all actual frame lengths, eligible six-frame history counts, per-record/sorted particle counts, minimum/maximumN, type IDs and auxiliary presence. Frame count301 is checked against metadata but no count or type is assumed. The existing128MiB payload guard remains an explicit preserved-failure boundary, not a known dataset maximum. Full arrays/records can guide a separate 3D trainer and budget after root review.

Sixteen tiny synthetic converter tests and seven mocked HTTP acquisition tests passed. They cover preservation of the third coordinate and auxiliary bits, rejection of 2D bytes, complete-source/record corruption, duplicate refusal, failure receipts/partial bytes, exact generation requests and no default network access. These are implementation checks, not Goop-3D observations. Pins and checks are recorded in `goop3d_source_preparation_v1.json`. Independent review is requested from the statistics agent; root should use the final review with these exact source hashes before launch.

## Concrete root commands

The prior second-VM repository is `/root/repos/AdaptGNS-cuda-20261006`; use its existing `.venv/bin/python`. Verify the path, free disk, source hashes and absence of an existing matching acquisition/conversion process or output before using these commands. The destinations below are fresh attempt names; preserve an already existing directory and review it rather than removing it. Copy the two new scripts into `cuda_preparation/` byte-for-byte; the exact generic reader is already expected at `research/prepare_full_waterdrop.py` but must be checked. This work uses CPU/network/storage only; it need not wait for an idle GPU.

After root reviews the source and concrete operation, acquire:

```sh
/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python -B -u \
  /root/repos/AdaptGNS-cuda-20261006/cuda_preparation/acquire_goop3d_train_valid.py \
  --execute \
  --output-dir /root/repos/AdaptGNS-cuda-20261006/goop3d_source_train_valid_20261006_v1
```

Save stdout/stderr and process identity outside the fresh output directory. Success requires `acquisition_report.json` status `complete` with all three completed checked objects. The receipt contains newly measured whole-object SHA256 values; no anticipated training-source SHA is invented here. Generation or CRC mismatch fails the attempt and must not silently update these pins.

After root verifies that completed receipt and unchanged converter/reader, convert:

```sh
/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python -B -u \
  /root/repos/AdaptGNS-cuda-20261006/cuda_preparation/prepare_goop3d_official.py \
  --execute \
  --input-dir /root/repos/AdaptGNS-cuda-20261006/goop3d_source_train_valid_20261006_v1 \
  --output-dir /root/repos/AdaptGNS-cuda-20261006/goop3d_numeric_train_valid_20261006_v1 \
  --acquisition-report /root/repos/AdaptGNS-cuda-20261006/goop3d_source_train_valid_20261006_v1/acquisition_report.json \
  --reader /root/repos/AdaptGNS-cuda-20261006/research/prepare_full_waterdrop.py \
  --splits train valid
```

Review the final structural report, both manifest hashes, observed counts/maxN/types/auxiliary contract and retained failures before designing training. Complete stored trajectories imply H=T−6; if allT301 thenH295. The released TensorFlow convenience evaluator drops the final frame and usesH294, so later evaluation must explicitly distinguish those conventions. No model or 3D test access is released by this preparation.
