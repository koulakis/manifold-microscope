# Development checks

Use Python 3.10 or newer and activate your development environment before running checks or committing.
Install the development tools, together with the dependencies needed for the experiment modules:

```sh
python -m pip install -e ".[benchmark,dev]"
python -m pre_commit install
python -m pre_commit run --all-files
```

The hooks run Ruff and Pyright across the two Python packages. They use the active environment so Pyright can resolve
the same scientific dependencies used to run the code. They do not create separate environments or install packages
when a commit is made. Run the same checks directly with:

```sh
python -m ruff check .
python tools/check_types.py
```

In the shared maintenance workspace, select the existing `microscope` pyenv environment. For a noninteractive check:

```sh
PYENV_VERSION=microscope pyenv exec python -m pre_commit run --all-files
```

Ruff checks common Python errors, 120-character lines, function annotations, function docstrings and documented
arguments in Google-style `Args` sections. Tests are exempt from annotation/docstring and line-length rules; generated
symbolic formulas are exempt from line-length checks. Pyright uses its basic mode for production code and the check
runner. The solver retains a narrow import-order exemption because it configures warning filtering before importing
training dependencies.

## Existing typing issues

`pyright-baseline.json` records the 176 existing diagnostics found when these checks were introduced. The type-check
runner passes the current Python interpreter explicitly to Pyright, compares diagnostic paths/rules/messages and
counts against that file, and fails on new diagnostics. Source line movement does not change a diagnostic's identity.
Missing dependencies always fail, including when refreshing the baseline. Resolved diagnostics also require a baseline
refresh so the recorded debt stays current.

Review raw Pyright output before accepting any baseline changes. To run the underlying checker directly in an activated
environment, use `python -m pyright --project pyproject.toml`. After reviewing its output, refresh with:

```sh
python tools/check_types.py --update-baseline
```

Review the resulting JSON diff and commit it with the corresponding fix or deliberate tooling/dependency upgrade.
Do not regenerate the baseline merely to make a new type error pass. Diagnostic details can change with scientific
dependency versions, so use the same development environment when comparing results.

## Style conventions

The hooks check code without rewriting it. Ruff's formatter is deliberately not a hook: its default collapsing of
short signatures/calls would undo the chosen multiline layouts. Keep signatures with three or more arguments on
separate lines, expand crowded calls, include abstract tensor/array shapes in argument docs, and put helpers before
callers. These layout, shape-description and ordering conventions still need review.

Tool versions are pinned in the `dev` extra in `pyproject.toml`; update that extra and reinstall the tools together when
upgrading. pre-commit 4.6.2 supports the project's Python 3.10 baseline. The Pyright `nodejs` extra supplies its Node.js
runtime within the Python environment without requiring a separate system installation. See the [Ruff configuration documentation](https://docs.astral.sh/ruff/configuration/),
[Pyright configuration documentation](https://github.com/microsoft/pyright/blob/main/docs/configuration.md) and
[pre-commit local-hook documentation](https://pre-commit.com/#repository-local-hooks) for the supported settings.
