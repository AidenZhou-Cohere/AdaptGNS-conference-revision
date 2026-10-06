# Root command sequence for the two B preflights

This is an inert operator supplement to candidate manifest `72cfe26b2d8c13ea50d24b8d85e450ef228a4bf9ef681fb18a2de525f8640e6d`. It does not authorize a run. The original 900-second source phase remains incomplete. The script uses only the two fresh amendment outputs and the unchanged scientific preparer.

Use the workspace virtual environment and the saved operator `work/deadline_research_20261005/cuda_preparation/sand_B_preflight_amendment_operator_code_audit_v1.py`. All issued local files go to `work/deadline_research_20261005/cuda_preparation/sand_B_preflight_amendment_released_root_v1`. That directory does not exist in the proposal.

After the human explicitly approves, save the exact human text to an ordinary local text file. Record the actual message reference and observation time. The following command arguments are the only human evidence values to substitute; the script resolves every fixed source/control path and hash itself.

```sh
work/venv/bin/python -B work/deadline_research_20261005/cuda_preparation/sand_B_preflight_amendment_operator_code_audit_v1.py record-approval --message-file HUMAN_MESSAGE_FILE --message-reference HUMAN_MESSAGE_REFERENCE --observed-utc ACTUAL_OBSERVATION_UTC
work/venv/bin/python -B work/deadline_research_20261005/cuda_preparation/sand_B_preflight_amendment_operator_code_audit_v1.py precheck
work/venv/bin/python -B work/deadline_research_20261005/cuda_preparation/sand_B_preflight_amendment_operator_code_audit_v1.py issue
```

The one shared 600-second phase starts at the earlier of the original precheck transport start and its host clock sample, so the precheck itself consumes this allowance. Both issued operations use its unchanged stop. The approval record must precede the start. Issuance checks the complete subsequent 600 + 11,760 + 3,600-second reservation and a five-second clock margin against 04:00 UTC.

Review the actual issued phase, two mode releases, two operational releases, and direct bootstrap argv. Then stage the exact two new source files, five immutable history records, and six actual control files in one bounded call:

```sh
work/venv/bin/python -B work/deadline_research_20261005/cuda_preparation/sand_B_preflight_amendment_operator_code_audit_v1.py stage
```

Run these as two separate original tool calls, optionally concurrently. Retain their original tool session IDs and observe each original exit; do not mistake a later observer exit for the original exit.

```sh
work/venv/bin/python -B work/deadline_research_20261005/cuda_preparation/sand_B_preflight_amendment_operator_code_audit_v1.py run --operation preflight_valid
work/venv/bin/python -B work/deadline_research_20261005/cuda_preparation/sand_B_preflight_amendment_operator_code_audit_v1.py run --operation preflight_test
```

For each original tool call, record its actual session ID, exit code and observation timestamp, then collect that operation's exact scalar files and timeout/owner/child absence proof:

```sh
work/venv/bin/python -B work/deadline_research_20261005/cuda_preparation/sand_B_preflight_amendment_operator_code_audit_v1.py record-original-exit --operation preflight_valid --session-id VALID_ORIGINAL_SESSION --exit-code VALID_ORIGINAL_EXIT --observed-utc VALID_EXIT_OBSERVATION_UTC
work/venv/bin/python -B work/deadline_research_20261005/cuda_preparation/sand_B_preflight_amendment_operator_code_audit_v1.py record-original-exit --operation preflight_test --session-id TEST_ORIGINAL_SESSION --exit-code TEST_ORIGINAL_EXIT --observed-utc TEST_EXIT_OBSERVATION_UTC
work/venv/bin/python -B work/deadline_research_20261005/cuda_preparation/sand_B_preflight_amendment_operator_code_audit_v1.py collect --operation preflight_valid
work/venv/bin/python -B work/deadline_research_20261005/cuda_preparation/sand_B_preflight_amendment_operator_code_audit_v1.py collect --operation preflight_test
```

After both successful collections, the independent scalar review helper consumes only those captured files and opaque original numeric hash evidence:

```sh
work/venv/bin/python -B work/deadline_research_20261005/cuda_preparation/review_completed_sand_B_preflight_amendment_code_audit_v1.py preflight_valid
work/venv/bin/python -B work/deadline_research_20261005/cuda_preparation/review_completed_sand_B_preflight_amendment_code_audit_v1.py preflight_test
```

Every issue, stage, run, original-exit observation, collection and review must finish before this single stop. A failed step ends admission: retain its local stdout/stderr/exit records and all remote bytes, and report the amendment incomplete. The success collector deliberately refuses failed original invocations; missing native-closure proof never becomes success. Do not retry, move the stop, fill an original never-launched output path, or shrink the final comparison.

Successful completion still requires root review and a separate final GPU release. The final 24-stage candidate must use the new B preflight paths, preserve its superseded original candidate, and retain unchanged scientific argv, model/data/source hashes and quotas. The operator supplies no GPU execution authority.
