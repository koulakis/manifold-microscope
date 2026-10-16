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


def test_success_is_marked_only_after_expected_outputs_exist(tmp_path):
    output = tmp_path / "experiment"

    def action(directory):
        result = directory / "result.txt"
        result.write_text("complete")
        assert not (directory / "COMPLETED").exists()
        return [result]

    assert run_experiment(output, action)
    assert (output / "COMPLETED").is_file()


def test_unfinished_prerequisite_skips_without_creating_output(tmp_path, capsys):
    output = tmp_path / "experiment"

    def action(directory):
        raise AssertionError("An experiment with unfinished inputs must not run.")

    assert not run_experiment(output, action, requires=[tmp_path / "input"])
    assert "unfinished inputs" in capsys.readouterr().out
    assert not output.exists()
