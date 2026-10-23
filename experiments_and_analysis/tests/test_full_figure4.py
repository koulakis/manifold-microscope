"""The full recipe retains every paper size and all three repetitions."""
from collections import Counter
from pathlib import Path

from experiments_and_analysis.config import load_config
from experiments_and_analysis.scripts.sweep import expand


def test_full_workload_and_seeds():
    path = Path(__file__).parents[1] / 'scripts/configs/paper/figure4.yaml'
    jobs = expand(load_config(path))
    fits = [job for job in jobs if job['model']['name'] == 'beta_vae']
    mmls = [job for job in jobs if job['model']['name'] == 'mmls']
    assert len(fits) == len(mmls) == 39
    expected = {f: 3 for f in [.01, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]}
    assert Counter(job['split']['train_fraction'] for job in fits) == expected
    assert Counter(job['training']['seed'] for job in fits) == {439: 13, 437875: 13, 473959: 13}
    assert all(job['analyses'][0]['name'] == 'errors' for job in jobs)
