"""The full recipe must retain every paper size and all three repetitions."""
from collections import Counter

from experiment_scripts.reproduction import full_figure4


def test_full_workload_and_seeds(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(full_figure4, 'stage', lambda *args: calls.append(args))
    full_figure4.reproduce(tmp_path, device='cpu', projection_method='original')
    fits = [c for c in calls if c[0] == 'fitting' and c[2] == 'vae']
    mmls = [c for c in calls if c[0] == 'fitting' and c[2] == 'mmls']
    assert len(fits) == len(mmls) == 39
    assert len([c for c in calls if c[0] == 'evaluation']) == 39
    assert len(calls) == 120
    expected = {f: 3 for f in [.01, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]}
    assert Counter(c[c.index('--fraction') + 1] for c in fits) == expected
    assert Counter(c[c.index('--seed') + 1] for c in fits) == {439: 13, 437875: 13, 473959: 13}
    assert all(c[c.index('--projection-method') + 1] == 'original' for c in mmls)
