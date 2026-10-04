import numpy as np
import pytest

from research.risk_benefit import aggregate_frames, frame_metrics


def test_error_rank_need_not_be_benefit_rank_and_sign_is_explicit():
    metrics, benefit = frame_metrics([1., 2., 3.], [1., 4., 9.], [0., 4., 12.])
    np.testing.assert_array_equal(benefit, [1., 0., -3.])
    assert metrics["risk_error_spearman"] == 1.
    assert metrics["risk_benefit_spearman"] == -1.
    assert metrics["fraction_harmful"] == pytest.approx(1/3)
    assert metrics["fraction_unchanged"] == pytest.approx(1/3)
    assert metrics["mean_signed_benefit"] == pytest.approx(-2/3)


def test_trajectory_aggregation_does_not_pool_frames_or_particles():
    records = [{"trajectory": "many", "metrics": {"mean_signed_benefit": 2.}},
               {"trajectory": "many", "metrics": {"mean_signed_benefit": 4.}},
               {"trajectory": "one", "metrics": {"mean_signed_benefit": 9.}}]
    result = aggregate_frames(records)
    assert result["trajectories"]["many"]["metrics"]["mean_signed_benefit"] == 3.
    assert result["equal_trajectory_mean"]["metrics"]["mean_signed_benefit"] == 6.


def test_undefined_correlation_and_nonfinite_data():
    metrics, _ = frame_metrics([1., 1.], [1., 2.], [1., 2.])
    assert metrics["risk_error_spearman"] is None
    assert metrics["risk_benefit_spearman"] is None
    assert metrics["fraction_unchanged"] == 1.
    with pytest.raises(ValueError):
        frame_metrics([np.nan], [1.], [0.])
