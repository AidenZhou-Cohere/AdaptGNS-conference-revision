# Sand runtime migration: separately reviewed owner scheduling amendment

The original recovery training attempt under release 42b132adc9b53aa090e9cae118992952f211b0ba1f79c64dcc950d86cafa91d9 exited unsuccessfully. All four workers were interrupted while loading their unchanged copied original 50,000-step checkpoint, before restoration or any optimizer update. Its complete outputs, logs, identities, controls and hashes remain retained.

The rejected process row was not retained in that attempt. An overlap between the owner's asynchronous nvidia-smi subprocess fork/exec window and process scanning is a source-supported hypothesis, not a proven identification of the original offending process.

Separate owner version 2 preserves every strict ownership predicate. It scans only after the prior GPU query future has completed, returned and been consumed; the next GPU query starts after the scan. No descendant or unknown process is exempted. A rejected row and full inventory are written before raising. This changes owner scheduling and diagnostics only. The original owner version 1 remains immutable.

A new attempt requires a separate root release after independent source, synthetic regression, failed-attempt and native-closure reviews. It uses fresh version-2 output/owner directories and restores the original immutable 50,000-step parents again. It never resumes or overwrites the failed recovery attempt. No automatic retry is enabled, and no replay is repeated or promoted. Original replay receipts remain bound.

The numerical trainer, adapter, endpoint auditor, all original source/data/model hashes, six-model six-policy study, 100,000-step endpoint, runtime contract, full training allowance 27404.671591931674 seconds, 15:05 UTC latest launch, 22:44 UTC training/audit stop, 18960-second downstream reserve and October 7 04:00 UTC analysis deadline remain unchanged. Missing fresh evidence or an expired launch window blocks a new attempt.
