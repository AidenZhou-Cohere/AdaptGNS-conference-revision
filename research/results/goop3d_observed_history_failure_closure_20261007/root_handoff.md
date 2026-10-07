# Root-only complementary publication plan

The sealed local addendum is
`work/d3_observed_failure_closure_addendum_UNADMITTED_20261007_v1/`, manifest
`7d8436e9f8d2550dcf0ba420a423d613a421c1c242900cf06f2562597e038db7`.
It complements the earlier immutable candidate and never edits that snapshot.

The fixed new destination is
`outputs/AdaptGNS/research/results/goop3d_observed_history_failure_closure_20261007`.
The original base publication must already exist exactly with manifest
`30d8b469aad2310c6abf0e9091d942b3dd9ad422aa24060c33a0d4a800fc4885`,
213 files, and copy-plan SHA
`3c96b82f8abcab42a4bb5fda7ac1a1ecf83f9995250eaa6ed0dcc2466c3f7fc7`.
The complementary helper checks those bytes directly and never calls the old
helper or repeats its preflight.

Root may run `python3 <this-directory>/copy_verify.py --copy` after independent
review. It requires branch `research/conference-revision` at
`b2517a4677dfc4dff9a31043da7f90f783d66943`, unchanged tracked files and no
unrelated untracked files. Root should install both result trees before copying
the separately reviewed repository README candidate, staging, committing or
pushing. The helper does not overwrite tracked files. If these preconditions
change, preserve this plan and prepare a reviewed successor rather than relaxing
the existing gate silently.

`--verify-installed` checks both exact result trees and permits later Git/README
changes. The installed helper accepts `--plan copy_provenance.json`. Default
mode is read-only. No mode of this new helper was executed during preparation;
the plan's source and hash checks are recorded separately as local static
packaging checks, not as a publication preflight or scientific validation.

Only the new result directory is exclusively created. Existing/partial output
is never replaced or deleted. Limits are 150 source files and less than 5 MB.
Root retains final review, actual publication, commit and push authority. The
report PDF is unchanged and reference-only. No science, compiler, source tests,
clock, live process probe, process action, retry or phase reset is part of this
plan.
