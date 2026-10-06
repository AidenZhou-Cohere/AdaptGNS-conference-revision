import pathlib,json,hashlib,shutil,datetime
W=pathlib.Path('/Users/aiden.zhou/Documents/Codex/2026-10-04/why-cangmai');P=W/'work/deadline_research_20261005/cuda_preparation';repo=W/'outputs/AdaptGNS';D=repo/'research/results/runtime_migration_and_d3_release_preparation_20261006';prior=repo/'research/results/scoped_execution_completion_preparation_20261006'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();ip=P/'recovery_d3_packaging_inventory_code_audit_v1/inventory_v2.json';assert sha(ip)=='9e2a10935abed1bbca35f4fb331a024071e80d1de9683bfe5741c1d2bbff0b7e';inv=json.loads(ip.read_text());assert not D.exists();D.mkdir()
registry=json.loads((prior/'dependency_paths.json').read_text())['files'];deps={}
for rel,m in registry.items():
 source=(prior/m['published_path']).resolve();assert sha(source)==m['sha256'];deps[rel]={'published_path':str(pathlib.Path('..')/source.relative_to(repo/'research/results')),'sha256':m['sha256'],'bytes':source.stat().st_size,'overlay':False}
for f in inv['files']:
 source=W/f['workspace_relative_path'];assert sha(source)==f['sha256'] and source.stat().st_size==f['bytes']
 if f['copy_recommendation']=='candidate_copy':
  dest=D/'workspace'/f['workspace_relative_path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);deps[f['workspace_relative_path']]={'published_path':str(dest.relative_to(D)),'sha256':f['sha256'],'bytes':f['bytes'],'overlay':True}
 else:
  source=repo/f['already_published_identical_paths'][0];assert sha(source)==f['sha256'];deps[f['workspace_relative_path']]={'published_path':str(pathlib.Path('..')/source.relative_to(repo/'research/results')),'sha256':f['sha256'],'bytes':f['bytes'],'overlay':False}
shutil.copyfile(ip,D/'source_inventory.json');(D/'dependency_paths.json').write_text(json.dumps({'files':deps},indent=2,sort_keys=True)+'\n')
(D/'README.md').write_text('''# Sand runtime migration and Goop 3D evaluation preparation

This supporting package preserves the unchanged numerical recovery adapter, both operational owner versions, endpoint auditor, CPU test sources, protocols and independent reviews. It also records the reviewed Goop 3D final-evaluation release and initial native startup. It contains no Sand or Goop 3D final accuracy result.

The four interrupted Sand models retain their original 50k parents and exact model/Adam/RNG lineage. All-four repeated restoration checks passed. The first recovery owner stopped before optimizer updates after rejecting a process whose identity was not recorded. Its source supports a possible own-query fork/exec race; the historical offender and original runtime-replacement cause remain unknown. The separate v2 owner serializes its GPU query and process inventory and records rejected rows, without weakening foreign-process checks. Both versions and the unsuccessful attempt review are retained. The two original seed-0 models reached 100k; the complete six-model cohort still gates evaluation.

The Goop 3D queue preserves all six 25k models, all 30 stages, both splits, six policies, H295 and every incomplete outcome. Its fixed operational allocation is not a complete-horizon runtime forecast. Original quota, per-stage cleanup and audit reserves remain unchanged; a separate 15-second outer emergency cleanup allowance is explicit in the review. Startup evidence does not certify final completion or accuracy.

## Scope and reconstruction

`source_inventory.json` gives original paths, sizes, hashes and the inclusion rule. `dependency_paths.json` resolves exact previously published workspace dependencies plus the 28 new files. These relative references require the surrounding research fork; this directory alone is not a self-contained raw experiment. The verification helper reconstructs that source layout into a fresh local directory for CPU synthetic tests and inert CLI checks. No dataset, checkpoint, GPU, SSH or credentials are required for those checks.

All original models, arrays, full replay receipts, failed stdout/stderr, transport/native inventories and execution controls remain preserved in their original local/remote study trees; they are uniformly excluded from this source-and-review package. The inventory explicitly records those exclusions. Live controls, SSH configuration, authentication and executable environments are excluded. Supporting reviews can contain historical paths and machine identities; this is an author research package, not an anonymized submission supplement.

The scientific sources and earlier fixed studies are immutable. This package grants no execution authority and does not authorize repeating training, replay, evaluation or failed capacity studies.
''')
print(D)
