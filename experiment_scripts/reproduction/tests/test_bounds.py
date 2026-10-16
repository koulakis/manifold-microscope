import numpy as np
import pytest

from experiment_scripts.reproduction.bounds import _curve


def test_curve_recovers_power_law_and_groups_repetitions():
    rows = [
        {"number_of_train_points": n, "hausdorff_distance": 2 * n ** -0.5}
        for n in (10, 10, 100, 100)
    ]

    curve, summary = _curve(rows, dimension=2)

    np.testing.assert_array_equal(curve["x"], [10, 100])
    np.testing.assert_allclose(curve["empirical"], 2 * curve["x"] ** -0.5)
    assert [len(values) for values in curve["empirical_all_repetitions"]] == [2, 2]
    assert summary["alpha"] == pytest.approx(0.5)
    assert summary["coefficient"] == pytest.approx(2.0)


@pytest.mark.parametrize("rows", [
    [{"number_of_train_points": 10, "hausdorff_distance": 0.1}],
    [
        {"number_of_train_points": 10, "hausdorff_distance": 0.1},
        {"number_of_train_points": 20, "hausdorff_distance": 0.0},
    ],
])
def test_curve_rejects_data_that_cannot_be_fit_in_log_space(rows):
    with pytest.raises(ValueError):
        _curve(rows, dimension=2)
