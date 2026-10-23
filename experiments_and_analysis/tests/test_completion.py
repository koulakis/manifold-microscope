"""Partial outputs are preserved, and completion requires all declared files."""
import pytest

from experiments_and_analysis.outputs import begin, finish, is_complete


def test_partial_directory_is_preserved_and_stops(tmp_path):
    partial = tmp_path / 'partial'
    partial.mkdir()
    result = partial / 'result.txt'
    result.write_text('retained partial result')
    with pytest.raises(RuntimeError, match='Incomplete output preserved'):
        begin(partial, {})
    assert result.read_text() == 'retained partial result'
    assert not (partial / 'COMPLETED').exists()


def test_success_is_marked_only_after_expected_outputs_exist(tmp_path):
    output = tmp_path / 'experiment'
    assert begin(output, {'seed': 7})
    with pytest.raises(RuntimeError, match='Missing stage outputs'):
        finish(output, ['result.txt'])
    assert not is_complete(output)
    (output / 'result.txt').write_text('complete')
    finish(output, ['result.txt'])
    assert is_complete(output)
    assert not begin(output, {'seed': 7})
    with pytest.raises(ValueError, match='mismatch'):
        begin(output, {'seed': 8})


def test_unfinished_prerequisite_stops_without_creating_output(tmp_path):
    output = tmp_path / 'experiment'
    with pytest.raises(RuntimeError, match='Unfinished inputs'):
        begin(output, {}, requires=[tmp_path / 'input'])
    assert not output.exists()
