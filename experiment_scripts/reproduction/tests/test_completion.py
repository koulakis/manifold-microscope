"""Partial outputs must remain untouched even when warnings are filtered."""
import warnings

from experiment_scripts.reproduction.common import run_experiment


def test_partial_directory_warns_and_skips(tmp_path, capsys):
    partial = tmp_path / "partial"
    partial.mkdir()
    result = partial / "result.txt"
    result.write_text("retained partial result")

    def action(directory):
        raise AssertionError("A partial experiment must not be executed.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert not run_experiment(partial, action)
    assert "WARNING" in capsys.readouterr().out
    assert result.read_text() == "retained partial result"
    assert not (partial / "COMPLETED").exists()
