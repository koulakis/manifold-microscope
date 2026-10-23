"""Sparse repetition selections are explicit rather than inferred from names."""
from experiments_and_analysis.scripts.sweep import expand


def test_separately_run_repetitions_are_both_selected():
    jobs = expand(dict(base=dict(model=dict(name='beta_vae')), grid=dict(repetition=[
        dict(index=0, seed=439), dict(index=2, seed=473959)])))
    assert [job['training']['trial'] for job in jobs] == [0, 2]
    assert [job['training']['seed'] for job in jobs] == [439, 473959]
