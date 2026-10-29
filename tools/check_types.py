"""Check for Pyright regressions against the explicitly recorded typing debt."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys


def diagnostic_key(diagnostic: dict, root: Path) -> tuple[str, str, str]:
    """Identify a diagnostic independently of source line numbers.

    Args:
        diagnostic: Pyright diagnostic containing a source path, rule and message.
        root: Repository root used to make paths portable.
    """
    return (
        Path(diagnostic['file']).relative_to(root).as_posix(),
        diagnostic.get('rule', ''),
        diagnostic['message'],
    )


def main(update_baseline: bool = False) -> int:
    """Run Pyright in the current Python environment and reject new diagnostics.

    Args:
        update_baseline: Explicitly replace the baseline after reviewing the raw Pyright diagnostics.
    """
    root = Path(__file__).resolve().parent.parent
    baseline_path = root / 'pyright-baseline.json'
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'pyright',
            '--pythonpath',
            sys.executable,
            '--project',
            str(root / 'pyproject.toml'),
            '--outputjson',
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        print(result.stdout, end='')
        print(result.stderr, end='', file=sys.stderr)
        return result.returncode
    report = json.loads(result.stdout)
    diagnostics = report['generalDiagnostics']
    # Missing dependencies are environment failures, never accepted typing debt.
    missing = [d for d in diagnostics if d.get('rule') in ('reportMissingImports', 'reportMissingModuleSource')]
    if missing:
        for diagnostic in missing:
            print(f"{diagnostic['file']}: {diagnostic['message']}")
        print('Run checks in the development environment with the package and benchmark dependencies installed.')
        return 1
    current = Counter(diagnostic_key(d, root) for d in diagnostics)
    if update_baseline:
        entries = [dict(file=file, rule=rule, message=message, count=count)
                   for (file, rule, message), count in sorted(current.items())]
        baseline_path.write_text(json.dumps(entries, indent=2) + '\n')
        print(
            f'Recorded {sum(current.values())} existing Pyright diagnostics. '
            'Review the baseline diff before committing.'
        )
        return 0
    entries = json.loads(baseline_path.read_text())
    known = Counter({(d['file'], d['rule'], d['message']): d['count'] for d in entries})
    added = current - known
    resolved = known - current
    for diagnostic in diagnostics:
        key = diagnostic_key(diagnostic, root)
        if added[key]:
            line = diagnostic['range']['start']['line'] + 1
            print(f'{key[0]}:{line}: {key[1]}: {key[2]}')
            added[key] -= 1
    new_count = sum((current - known).values())
    if resolved:
        print(f'{sum(resolved.values())} baseline diagnostics resolved; refresh and review the reduced baseline.')
    print(f'Pyright: {new_count} new diagnostics; {sum(known.values())} recorded baseline diagnostics.')
    return 1 if new_count or resolved else 0


if __name__ == '__main__':
    # Keep this development bootstrap independent of optional CLI dependencies.
    if sys.argv[1:] not in ([], ['--update-baseline']):
        print('Usage: python tools/check_types.py [--update-baseline]', file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(update_baseline=sys.argv[1:] == ['--update-baseline']))
