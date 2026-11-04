"""Render synthetic curves without saving figures; inspect actual band regions."""
from typing import cast

import matplotlib

matplotlib.use('Agg')
import numpy as np
import pytest
from matplotlib import pyplot as plt

from experiments_and_analysis.plotting import bounds as style


@pytest.mark.parametrize('counts,has_band', [
    ([1, 1, 1, 1], False), ([2, 2, 2, 2], False),
    ([3, 3, 3, 3], True), ([3, 3, 1, 3], True),
])
def test_band_requires_three_repetitions(tmp_path, monkeypatch, counts, has_band):
    values = [list(np.arange(n) + 1.) for n in counts]
    data = dict(mmls=dict(x=np.array([5., 20., 60., 100.]),
                          empirical=np.array([np.mean(v) for v in values]),
                          empirical_all_repetitions=values))
    monkeypatch.setattr(plt, 'savefig', lambda *a, **k: None)
    close = plt.close
    monkeypatch.setattr(plt, 'close', lambda *a, **k: None)
    try:
        style.create_repetitions_shaded_plot(data, tmp_path)
        ax = plt.gcf().axes[0]
        # This renderer supplies numeric arrays and string labels; Matplotlib's stubs are broader.
        y_data = cast(np.ndarray, ax.lines[0].get_ydata())
        label = cast(str, ax.lines[0].get_label())
        np.testing.assert_allclose(y_data, [np.mean(v) for v in values])
        assert bool(ax.collections) == has_band
        assert ('σ' in label) == has_band
        if counts == [3, 3, 1, 3]:
            # No polygon may shade across the point with only one repetition.
            for polygon in ax.collections[0].get_paths():
                vertices = cast(np.ndarray, polygon.vertices)
                x = vertices[:, 0]
                assert not (x.min() < 60 < x.max())
    finally:
        close('all')
