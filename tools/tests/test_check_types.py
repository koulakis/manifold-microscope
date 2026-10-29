"""Protect the pre-commit type check's handling of existing and new diagnostics."""
import json
from pathlib import Path
import subprocess

import pytest

from tools import check_types


def configure_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    rule: str = 'reportAssignmentType',
    returncode: int = 1
) -> Path:
    """Supply controlled checker output and one previously recorded diagnostic."""
    monkeypatch.setattr(check_types, '__file__', str(tmp_path / 'tools' / 'check_types.py'))
    baseline = tmp_path / 'pyright-baseline.json'
    message = 'Type str cannot be assigned to int'
    baseline.write_text(json.dumps([dict(file='example.py', rule=rule, message=message, count=1)]))
    diagnostic = dict(
        file=str(tmp_path / 'example.py'),
        rule=rule,
        message=message,
        range={'start': {'line': 99}},
    )
    result = subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=json.dumps({'generalDiagnostics': [diagnostic] * count}),
        stderr='tool failure' if returncode == 2 else '',
    )
    monkeypatch.setattr(check_types.subprocess, 'run', lambda *args, **kwargs: result)
    return baseline


@pytest.mark.parametrize(('count', 'expected'), [(0, 1), (1, 0), (2, 1)])
def test_diagnostic_counts_and_line_movement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    expected: int
) -> None:
    configure_report(tmp_path, monkeypatch, count)
    assert check_types.main() == expected


def test_explicit_update_records_duplicate_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = configure_report(tmp_path, monkeypatch, 3)
    assert check_types.main(update_baseline=True) == 0
    assert json.loads(baseline.read_text())[0]['count'] == 3
    assert check_types.main() == 0


@pytest.mark.parametrize('rule', ['reportMissingImports', 'reportMissingModuleSource'])
def test_missing_dependencies_cannot_enter_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rule: str
) -> None:
    baseline = configure_report(tmp_path, monkeypatch, 1, rule=rule)
    original = baseline.read_bytes()
    assert check_types.main(update_baseline=True) == 1
    assert baseline.read_bytes() == original


def test_checker_failures_preserve_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = configure_report(tmp_path, monkeypatch, 0, returncode=2)
    original = baseline.read_bytes()
    assert check_types.main(update_baseline=True) == 2
    assert baseline.read_bytes() == original
