# Supplement packaging and asset attribution audit

Checked October 4, 2026 using a tracked working-tree inventory at repository HEAD `1567ebbf2a76a3b54264f5b6cb506c16418360e9`, the saved official AISTATS 2027 CFP/FAQ, and the official GNS release sources described below. Each inventoried file has its own byte hash; concurrent work is not certified by the HEAD value. **The public fork and current evidence report are not ready to upload as an anonymous supplement.** The initial audit created no archive; a later source-only author-review bundle is documented below. Frozen training code, protocols and original measurements were not changed.

## Concrete findings

The root `AdaptGNS/LICENSE` already applies an MIT notice to the software and associated documentation. It names “Copyright (c) 2026 Aiden Zhou” and the Geoelements GNS Authors. The new research code is therefore not missing a root license file. The notice also directly identifies the author. The MIT terms require inclusion of the copyright and permission notices in copies or substantial portions of the software; blanket deletion of license files is not an appropriate anonymization step. An author-controlled anonymous distribution notice, or another permitted packaging arrangement, needs deliberate author verification while retaining third-party notices. This audit does not determine who can authorize changes to every contributed file.

An inventory of **293 tracked files, 65,614,347 bytes**, found name, path, cluster or email indicators in **53 files**. These are audit flags, not 53 independent violations: third-party attribution and contact information must be distinguished from first-party identity. In particular:

| Surface | Verified identifying content | Packaging action |
|---|---|---|
| Root `LICENSE`, line 3 | First-party author name | Resolve the license/anonymity presentation with the rights holder; preserve required third-party notices. |
| Root `README.md`, line 5 | `aidenzhou8/AdaptGNS` and upstream commit | Write a separate anonymous reproduction README. Keep the public development README intact. |
| `README_legacy_release.md` and two legacy plotting/counting script docstrings | Yale/Misha cluster references | Exclude unnecessary historical execution scripts; retain their scientific results and provenance in an appropriate reviewed form. |
| All 20 historical `Sand/models/**.json` and `WaterDrop/models/**.json` files | Cluster paths containing username `az474` | Make a separate metadata view with logical relative paths. Preserve original evidence and original hashes. |
| All 18 `jobs/*.sh` scripts | Personal cluster paths and executable locations | Exclude from the minimal local reproduction package; they are not the fixed local protocol. |
| `research/results/{benefit_head,risk_benefit,rollout_summary,full_rollout_smoke,metal_cli_smoke}.json` | Absolute paths under `/Users/aiden.zhou/` | Review JSON fields and command strings; emit a separate portable metadata view. Do not overwrite the measured originals. |
| `adaptive-gns/AUTHORS.md`, contributor policies and two upstream SLURM scripts | Geoelements author contacts | These are third-party attribution, not evidence of the submitting author's identity. Keep required notices; omit irrelevant policies and upstream jobs from the minimal package. |
| Public fork URL | GitHub account identifies author | Do not use the current fork URL as the anonymous supplement link. The conference permits a ZIP upload. |
| `revision_report.pdf`, page 1 | Printed `github.com/AidenZhou-Cohere/AdaptGNS-conference-revision` | Keep this author-facing evidence report out of the submitted anonymous package, or prepare a separately reviewed submission version. Its neutral PDF Author field does not anonymize its body. |

The tracked inventory contains **no `.pt`, `.pth`, `.ckpt` or `.tfrecord` files** and no private reviewer PDFs. All **65 tracked NPZ files** loaded without enabling pickle; their array names, shapes, dtypes and string fields were inspected. This is a content-format check, not permission to redistribute dataset-derived arrays.

The six tracked PDF figures and the evidence report were checked with `pypdf` for text, document metadata, URI annotations, attachments and XMP presence. None had URI annotations, attachments or XMP metadata. The six figure PDFs had no matches to the checked first-party name/path patterns. Some figure Author fields read `AdaptGNS research audit`; this is project branding and still merits a final human anonymity review. The native-editor manuscript PDF was not available as an exported file for this audit, so its metadata, actual main-text pagination and final visual anonymity remain unverified. The source author field is anonymous.

## Asset terms and attribution

| Asset | Evidence and actual terms checked | Remaining action |
|---|---|---|
| First-party AdaptGNS code and associated documentation | Root `LICENSE`: MIT; first-party 2026 and Geoelements 2021 copyright notices; permission/warranty text present. | Confirm contributor rights and the first-party license presentation for anonymous review. Do not silently strip notices. |
| Geoelements GNS port | `adaptive-gns/license.md`: MIT, “Copyright (c) [2021] [Geoelements GNS Authors]”. `AUTHORS.md` and `CITATION.cff` are present. | Keep the upstream copyright and permission notice. Retain authorship/citation provenance; Kumar and Vantassel (JOSS 2023, doi:10.21105/joss.05025) is already in the manuscript bibliography. No comprehensive historical derivation audit is claimed. |
| Official DeepMind GNS software | Official parent repository `LICENSE`: Apache License 2.0. The saved official `download_dataset.sh` carries a DeepMind 2020 Apache header. The downloader is in `work/sources/`, not the tracked fork. | If redistributing that script or other Apache-covered source, include the license, preserve applicable notices, mark modified files, and preserve any applicable upstream NOTICE as required by section 4. The minimal package can use its own existing download commands instead of bundling the script. |
| Official WaterDrop/Sand datasets and metadata | The official `learning_to_simulate/README.md` identifies the storage URLs and asks code users to cite Sanchez-Gonzalez et al. (ICML 2020). No dataset-specific license statement was found in that README, parent README or the checked project page. The page's JavaScript license comments concern web software. | **Dataset license remains unverified.** Do not label these datasets CC-BY or Apache solely from the software license. Record official acquisition URLs and checksums, omit raw data from the supplement, and have the author establish the applicable dataset terms. The absence of a located statement is not a conclusion that research use is forbidden. |
| Numeric result arrays, derived figures and future model checkpoints | Produced from the local/released experiments; source/data/checkpoint hashes distinguish their origins. A root software notice alone does not settle every third-party data right or weight-distribution question. | Author to confirm the intended redistribution terms. Maintain clear historical/pilot/smoke/full-model labels and all unsuccessful outcomes. Checkpoint files are currently local and unbundled. |
| Python dependencies | `research/requirements-local-lock.txt` contains version pins, with no embedded personal wheel URLs or editable local paths. | Include installation instructions and the lock as an observed macOS environment record. Do not bundle the virtual environment or dependency source/binaries; doing so would require their own notice inventory. |
| AISTATS 2027 template | Official paper pack is saved locally; template provenance and modification comments are embedded in the standalone manuscript source. | Preserve relevant template attribution. Do not claim the whole template is covered by the project's MIT license. Verify final PDF and source packaging against the venue's instructions. |

Official sources checked:

- [AISTATS 2027 submission FAQ](https://virtual.aistats.org/Conferences/2027/SubmissionFAQ): “Submissions must not include author names, affiliations, acknowledgments, or links from which the authors' identities or institutions can be inferred.” All supplements share the full-paper deadline and may be a single PDF or ZIP.
- [DeepMind GNS README](https://github.com/google-deepmind/deepmind-research/blob/master/learning_to_simulate/README.md), [parent license](https://github.com/google-deepmind/deepmind-research/blob/master/LICENSE), [parent README](https://github.com/google-deepmind/deepmind-research/blob/master/README.md), and [project site](https://sites.google.com/view/learning-to-simulate). These were retrieved for attribution only; no test trajectories were read.

Local authoritative-source snapshots have SHA256 values:

```text
deepmind_learning_to_simulate_README.md  9db737244fccd3ea37d525ba244b7ea2bddd300f420a346bbfdd2893c419a69a
deepmind_research_LICENSE               cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30
deepmind_research_README.md             76424f9373d1265afb27e2e7551d5d1473a652d219506813846bf256523dbc8c
deepmind_learning_to_simulate_site.html c7a2890859c1c1faa0db70e607f81c7cb48f245078722988bfecfb5d9a54454b
```

## Minimum reproducible package

Use an explicit file allowlist assembled after the experiments and manuscript freeze. A minimal package would include:

1. An anonymous `README` with environment setup, relative data/result paths, source hierarchy, exact reproduction commands, expected outputs, device limitations, runtime, failures and remaining missing assets. Include the substantial AI-use disclosure in the appropriate manuscript location.
2. Required license/attribution material, plus a concise asset-provenance table. Resolve the identifying root notice before calling the package anonymous.
3. Unmodified `adaptive-gns/gns/*.py`, required evaluation scripts, all `research/*.py` needed for claims, `tests/`, `research/tests/`, the version lock and dependency setup. The final relevant suite must pass from the extracted package. Its exact command uses `PYTHONPATH="$PWD:$PWD/adaptive-gns"` and `python -m pytest tests research/tests -q`.
4. The immutable fixed protocol and official test-source identity; training/data/source/configuration/checkpoint hashes; numeric manifests with acquisition commands. Do not edit frozen training code or protocol to conceal paths. Dataset files are obtained separately from the official source under the verified applicable terms.
5. All evidence needed by the retained paper claims: original historical arrays; compact pilot seed files; benefit/risk coefficients and samples; both autonomous pilot horizons with failures; full-model records only when produced. Include all requested seed/policy/trajectory outcomes, missing runs and unsuccessful outcomes. Keep smoke checks separate. Provide deterministic summary/figure commands and a mapping from manuscript tables/figures to inputs.
6. A path/hash manifest for the distributed files. If metadata views are transformed for portability/anonymity, retain immutable original hashes and record distinct package hashes and the precise transformations. Validate dependent references against the packaged views. Do not claim a transformed JSON has the original byte hash or change source/data/checkpoint hashes to hide an inconsistency.
7. Any final-checkpoint weights necessary for immediate evaluation, with hashes, architecture/normalization metadata and verified distribution terms. If omitted for size or licensing, say explicitly that the package supports analysis/retraining but cannot reproduce checkpoint evaluation without training or separately supplied weights. Original historical checkpoints remain missing and must not be implied available.

Exclude `.git/`, remotes/history, raw data, converted raw trajectory arrays, all environments/caches, active queue directories, locks/PIDs, local logs, cluster jobs, unrelated upstream CI/policies, private reviewer PDFs, the author-facing README/report/continuation notes and this audit. Figures and scientific outcomes are not excluded merely because they are adverse. Do not place a new anonymous overlay back into the frozen run directories.

## Verification aid and open checks

`work/audit_supplement_candidates.py` is a read-only inventory aid. It reads tracked files, rejects paths outside the repository and symlinks, records hashes, checks text indicators, inspects numeric NPZ files without pickle, and lists broad candidate groups. It **does not create an archive, redact files, approve licenses, or certify anonymity**. Its candidate groups intentionally require review: they currently include identifying notices, some historical script docstrings and result metadata. Current output is `work/supplement_candidate_inventory.json`; the separate complete `pypdf` inspection is `work/supplement_pdf_inventory.json`. New untracked aggregation/monitoring code was being developed concurrently and is outside this snapshot. Rerun after the final commit.

Before submission: resolve license presentation and dataset terms; assemble a separate anonymous tree; verify every path/hash transformation; run tests and figure/table regeneration from that tree; inspect the exported PDF and archive metadata and all visible links; verify the registered abstract/title and human AI-verification statement; then obtain author review. No author verification, conference readiness, or final supplement compliance is asserted here.

## Post-commit inventory update

The read-only inventory was rerun after tested commit `6a272646fd3248f4a87cd131a056ab6e8870764a` was pushed. It now covers **313 tracked files, 66,475,743 bytes**, with the same **53 flagged files**. New same-state/aggregation/theory code and compact integration evidence introduced no additional matches to the checked identity/path patterns. The machine-readable snapshot is `work/supplement_candidate_inventory_6a27264.json`; the earlier inventory is preserved. This is not a fresh PDF visual/metadata audit or an anonymity certification.

The manuscript now passes a native compile-time assertion that its configured main text is at most eight pages. This narrows the pagination uncertainty but does not replace inspection of the final exported PDF, metadata, visible links or submission archive.

## Non-anonymous source-only review bundle

The versioned [author-review bundle](source_reproduction_bundle_6a27264_README.md) preserves 80 selected tracked files from commit `6a27264`, plus explicit build/provenance documentation. It retains all included source and attribution bytes, excludes every numeric scientific evidence file uniformly, and records a hash-only catalog of omitted evidence. No dataset, checkpoint, raw trajectory, log or reviewer document is included.

All payload hashes, 47 small synthetic checks, six CLI-help checks, six import-location checks and deterministic repacking passed in the extracted tree. The first failed builder attempt is preserved separately. These are portability checks in the existing local environment, not a clean-machine installation, full empirical reproduction, anonymity certification or final supplement. Its license and identity notices are deliberate and visible.

The refreshed [f3d0d24 bundle](source_reproduction_bundle_f3d0d24_README.md) is also complete. It preserves 87 tracked source/protocol/notice files and uniformly omits/catalogues all 184 tracked scientific evidence files. All 102 extracted-tree tests, eight CLI-help checks, nine import-location checks, 71 Python parses, payload hashes and deterministic repacking passed. The earlier bundle is unchanged. The same non-anonymity, missing empirical assets and unverified dataset-redistribution limits apply.
