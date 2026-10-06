# Prospective exact Sand downloader

`download_sand_public.py` is newly written and **has not been executed or network-tested by its author**. Root review and bounded tests are required before using it on the provided VM. It imports only Python's standard library.

After review, a root-operated invocation could be:

```sh
python download_sand_public.py --output /chosen/existing/parent/sand-v1-acquisition --files train.npz valid.npz --max-seconds 1800
```

The output directory must not already exist; the parent must exist. Metadata is always fetched first and matched to the already verified 363-byte public/local SHA256. Omitting `--files` requests all four exact files, including `test.npz`; selecting train/valid keeps the reserved test transfer separate. No array payload is decoded in either mode, and the utility never imports NumPy or invokes pickle.

The normal public project GET establishes a new in-memory CSRF cookie; the exact official download PUT requests one allowlisted file at a time. A separate opener follows the returned HTTPS link without portal cookies, credentials, CSRF or Referer. Both openers ignore environment proxies; redirects are limited to three per opener and exact allowed hosts. Login or unexpected hosts/statuses fail rather than triggering another access method. Archive download/auth behavior for the official PUT route is still unverified.

Each transfer streams to a fresh `.part`, checks the exact published byte count, records SHA256 and safe response headers, and then inspects metadata bytes or ZIP/NPY headers. Fully successful files receive their final names. On error, earlier completed files, `.part` files and a JSON report remain; there is no automatic retry, append, cleanup or overwrite of an old acquisition. Redirect capability URLs, cookies and raw exception text are not saved. Time is checked between network reads; an in-flight socket read can take up to its 30-second timeout beyond a check.

ZIP inspection bounds directory size/member count and total claimed uncompressed bytes, rejects unsafe/encrypted/unexpected members, and preserves central-directory order. NPY headers are parsed with `ast.literal_eval`; object dtypes are reported without unpickling. Array values, full-member CRCs, finiteness, semantic types and trajectory completeness are **not** validated. Unknown header/schema forms stop inspection rather than broadening the decoder. This utility does not convert data or qualify it for training.

Exit codes: **0** means requested files and structural inspection completed without object dtypes; **2** means they completed but object dtypes were detected; **1** means failed or not started. Neither 0 nor 2 establishes numeric-data validity or scientific readiness. Inspect `download_report.json`; preserve any failure outcome. The report includes the downloader hash, source/version, fixed limits and per-file acquisition/header evidence.
