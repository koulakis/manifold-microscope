"""Discover repetitions run separately, rather than assuming a contiguous grid."""
from experiment_scripts.manifold_fitting.paths import model_directories


def test_separately_run_repetitions_are_both_discovered(tmp_path):
    names = [f"custom_dsprites_balanced__beta_vae__1.0__4__0__{rep}" for rep in (0, 2)]
    for name in names + ["datasets", "custom_dsprites_balanced__MMLS__1.0__4__0__0"]:
        (tmp_path / name).mkdir()
    found = list(model_directories(tmp_path))
    assert [path.name for path, _, _ in found] == names
    assert {(dataset, dimension) for _, dataset, dimension in found} == {("custom_dsprites_balanced", 4)}
