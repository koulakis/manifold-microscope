"""A separately launched repetition must retain its selected seed and budget."""
from typer.testing import CliRunner

from experiment_scripts.manifold_fitting import training


def test_selected_training_repetition(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(training, "export_fixed_grid_test_set_and_rest_for_train", lambda **kwargs: None)
    monkeypatch.setattr(training, "train_model", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = CliRunner().invoke(training.app, [
        "--output-path", str(tmp_path), "--datasets", "custom_dsprites_balanced",
        "--models", "beta_vae", "--dimensions", "1", "--fractions", "0.5",
        "--repetitions", "2", "--seeds", "473959", "--device", "cpu",
        "--num-workers", "0", "--max-iter", "1", "--strict-budget"
    ])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0].name == "custom_dsprites_balanced__beta_vae__0.5__1__0__2"
    assert args[2] == 473959
    assert kwargs["device"] == "cpu"
    assert kwargs["num_workers"] == 0
    assert kwargs["model_overrides"] == {"max_iter": 1, "strict_budget": True}


def test_mmls_projection_option(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(training, "export_fixed_grid_test_set_and_rest_for_train", lambda **kwargs: None)
    monkeypatch.setattr(training, "train_model", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = CliRunner().invoke(training.app, [
        "--output-path", str(tmp_path), "--datasets", "custom_dsprites_balanced",
        "--models", "MMLS", "--dimensions", "1", "--fractions", "0.5",
        "--repetitions", "0", "--device", "cpu", "--projection-method", "thin_svd"
    ])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][1]["model_overrides"] == {"projection_method": "thin_svd"}
