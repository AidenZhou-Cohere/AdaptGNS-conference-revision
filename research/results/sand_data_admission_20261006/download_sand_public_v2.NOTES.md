# Sand downloader v2: exact official host correction

The first VM acquisition stopped before metadata bytes were transferred because the original host allowlist rejected the returned link. The original source and failed attempt remain preserved; this is a separate replacement candidate.

On 2026-10-06 at 01:27:58 UTC, a bounded local request visited the [public project](https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-3702), obtained its normal anonymous CSRF cookie, and made the captured official public download PUT for **metadata.json only**. HTTP 200 returned:

- Scheme: `https`
- Host: `designsafe-download01.tacc.utexas.edu`
- Port: unspecified, hence HTTPS default 443
- User information and fragment: absent

The link was not followed. No capability URL, cookie or payload was printed or retained. This is direct evidence from the normal authenticated-TLS public DesignSafe response that the missing host is its supplied download host; it is not an inferred wildcard or an authentication workaround. See `sand_download_host_probe_v2.json`.

`download_sand_public_v2.py` adds **only that exact host** to the download allowlist. Portal requests remain restricted to `www.designsafe-ci.org`; download requests remain in a separate opener without portal cookies, CSRF, Authorization or Referer. HTTPS, port, redirect, byte-count, hashing, fresh-directory, retained-partial and header-only inspection bounds remain in place. Acquisition schema is now `sand_public_acquisition_v2`.

The sole additional parser correction requires ZIP64 end-record payload size **44**, matching CPython `zipfile`'s fixed-record reader. Extensible sectors are rejected so the bounded precheck and `ZipFile` cannot parse different directory sizes. This conservatively narrows accepted input; it does not weaken the ZIP bounds. No pickle or array conversion is added.

Hashes:

- Preserved original: `e106d2e62b02679a610dd77e2286dad3c503750fd40b704b561a9f14a675c217`
- New v2: `864d9c974deabceb19b5f3b8e88ced1ea7281595a0d3a526e8d39f382aa4605a`

`sand_downloader_v2_change_record.json` records these changes. **V2 itself has not been executed or network-tested by this agent.** Root owns review, any VM invocation and its fresh output directory. For a bounded first check, the existing CLI supports `--files metadata.json`; default still requests all four files, and metadata is always first. Successful header inspection remains distinct from numeric validity or training admission.
