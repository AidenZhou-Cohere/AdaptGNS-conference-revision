# Independent full-rollout aggregation audit

Passed: **False**. Checks: 76; input files: 7.

Every saved compact record, raw scalar JSON record and aggregate remains preserved. No evaluator, production summarizer, model or dataset array was loaded.


Audit anomalies (retained):

```json
[
  {
    "type": "AttributeError",
    "message": "'NoneType' object has no attribute 'get'",
    "traceback": "Traceback (most recent call last):\n  File \"/Users/aiden.zhou/Documents/Codex/2026-10-04/why-cangmai/work/evaluation_completion_20261005/rollout_analysis.py\", line 230, in <module>\n    if raw[\"warmup\"].get(\"passes\"):\n       ^^^^^^^^^^^^^^^^^\nAttributeError: 'NoneType' object has no attribute 'get'\n"
  }
]
```
