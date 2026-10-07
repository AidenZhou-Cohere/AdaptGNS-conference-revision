# Saved scientific results

`cross_material/printed_statistic_map.json` contains the records shown in the main quantitative figures; `all_scalar_statistic_map.json` supplies the remaining paired effects, absolute values, physical diagnostics and descriptive cost. Records retain ordered seed values, complete-population means, sample SDs and nulls.

`tables/source_tables.json` contains saved decimal cells for every primary policy and both training arms, including all 34 primary and34 physical/cost rows. `goop3d/observed_statistics.json` contains the18 displayed observed-history contrasts. `goop2d/qualitative_glyphs.json` stores only the five displayed particle panels, including all 1,083 particles per panel and strict outside-box glyphs. These are saved figure coordinates, not a replacement for the simulation dataset.

`controls/action_benefit.json` preserves the WaterDrop residual-versus-action comparison, including negative action benefit. `controls/graph_conventions.json` retains the self-message/cap factorial contrasts and both current- and previous-observed risk controls. `controls/waterdrop_objective_rollouts.json` retains the separate faithful/NLL full-horizon comparison with uncapped no-self-loop evaluation and all 8 NLL coordinate-guard failures. `failure_accounting.json` defines the failure denominators and the 3 Goop candidate-guard cases. These controls remain separate from graph-exposure comparisons.

The standard-library table command copies saved cells and formats existing seed statistics. The figure command uses the same values and retains undefined means. Neither command trains a model, opens simulation arrays, pools additional samples or selects a checkpoint.
