"""Feed actual frozen D3 producers tiny CPU fixtures; no official data or CUDA."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_train_goop3d_graph_support_cuda_v2 import synthetic, entry as trainer

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('d3_scalar_producer_integration', HERE / 'summarize_goop3d_graph_support_v1.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)


@pytest.fixture
def sources(synthetic):
    E = S.load_scalar('evaluate_goop3d_graph_support_v1.py')
    native = E.load(HERE / 'goop3d_native_evaluation_v1.py', S.SOURCE_PINS['goop3d_native_evaluation_v1.py'], '_d3_scalar_tiny_native')
    return native, S.load_scalar('goop3d_diagnostic_metrics_v1.py')


@pytest.mark.parametrize('policy', ['base', 'laggedrisk25', 'relative-velocity-RMS25'])
def test_frozen_rollout_complete_or_guarded_row_accepted(synthetic, sources, policy):
    native, _ = sources; np = synthetic.np; model, _ = synthetic.make()
    points = synthetic.batch[0][:8, 0].numpy()
    positions = np.broadcast_to(points, (301, 8, 3)).copy()
    item = {'source_index': 0, 'trajectory_id': 'valid:000000', 'particles': 8, 'size_group': 'fixed_source_grid', 'policy': policy}
    with synthetic.torch.no_grad():
        row, arrays = native.rollout(model, positions, np.full(8, 7, dtype=np.int64),
            {'bounds': [[.1, .9]] * 3, 'default_connectivity_radius': .025}, policy, 295, 93000, 'cpu', trace_steps=(1, 10, 50, 200, 295))
    row.update(**item, arm='base', training_seed=0, objective='faithful', synchronized_call_seconds=10., publication_seconds=.1,
               mse_at_declared_trace_steps={str(k): row['mse_per_step'][k-1] if row['completed_steps'] >= k else None for k in (1, 10, 50, 200, 295)})
    S.validate_rollout(row, 'base', 0, item)
    assert arrays['predicted_positions'].shape[-2:] == (8, 3)


@pytest.mark.parametrize('mode', ['same-state', 'clean-validation'])
def test_frozen_diagnostic_rows_match_scalar_contract(synthetic, sources, mode):
    native, diagnostic = sources; np = synthetic.np; model, _ = synthetic.make()
    points = synthetic.batch[0][:8, 0].numpy(); positions = np.broadcast_to(points, (8, 8, 3)).copy()
    item = {'source_index': 0, 'target_frame': 7 if mode == 'same-state' else 6, 'schedule_index': 0, 'trajectory_id': 'valid:000000'}
    with synthetic.torch.no_grad():
        if mode == 'same-state':
            row, arrays = diagnostic.same_state(native, model, positions, np.full(8, 7, dtype=np.int64),
                                               {'bounds': [[.1, .9]] * 3}, item, 'valid', 0, 'cpu')
        else:
            helpers = SimpleNamespace(**vars(synthetic.helpers)); helpers.unpack_batch = lambda examples: trainer.unpack_batch(helpers, examples)
            row, arrays = diagnostic.clean_validation(native, helpers, model, positions, np.full(8, 7, dtype=np.int64), item, 'cpu')
    assert row['status'] == 'complete'
    row.update(arm='base', training_seed=0, objective='faithful', synchronized_call_seconds=10., publication_seconds=.1)
    S.validate_diagnostic(row, 'base', 0, item, 'valid', mode, 8)
