# Development checks

Use Python 3.10 or newer and activate your development environment before running checks or committing.
Install the development tools, together with the dependencies needed for the experiment modules:

```sh
python -m pip install -e ".[benchmark,dev]"
python -m pre_commit install
python -m pre_commit run --all-files
```

The hooks run Ruff and Pyright across the maintained Python code, including tests. They use the active environment so
Pyright can resolve the same scientific dependencies used to run the code. They do not create separate environments
or install packages when a commit is made. Run the same checks directly with:

```sh
python -m ruff check .
python -m pyright --project pyproject.toml --warnings
```

In the shared maintenance workspace, select the existing `microscope` pyenv environment. For a noninteractive check:

```sh
PYENV_VERSION=microscope pyenv exec python -m pre_commit run --all-files
```

Ruff checks common Python errors, 120-character lines, function annotations, function docstrings and documented
arguments in Google-style `Args` sections. Tests are exempt from annotation/docstring and line-length rules; generated
symbolic formulas are exempt from line-length checks. Pyright uses standard mode for all maintained Python files,
including tests, the root test configuration and the Sphinx configuration. Generated build outputs are excluded.
The solver retains a narrow import-order exemption because it configures warning filtering before importing
training dependencies.

## Dense grid reference tests

The dense basic, volume, curvature and reach reference tests require CUDA by default. They fail during setup if CUDA
is unavailable, before generating large input grids. Use `MICROSCOPE_TEST_DEVICE` to select another CUDA device or opt
into CPU execution:

```sh
MICROSCOPE_TEST_DEVICE=cpu python -m pytest microscope/computations_grid/tests/test_curvature.py -vv --durations=10
```

With CUDA available, run the same tests without the override:

```sh
python -m pytest microscope/computations_grid/tests/test_curvature.py -vv --durations=10
```

`test_curvature.py` runs intrinsic curvature only. The marked 3D/4D tests are skipped unless `--runslow` is supplied.
The full-grid total-curvature integral remains in the default selection and uses a 10,223 by 10,344 grid. To select
that integral explicitly:

```sh
python -m pytest microscope/computations_grid/tests/test_curvature.py -k total_scalar_curvature -vv
```

The separate extrinsic file contains reduced 2D analytic grids and the focused geometry regressions, without the
large slow 3D/4D cases. Its analytic grid tests target at most five seconds each on the development RTX 4090:

```sh
python -m pytest microscope/computations_grid/tests/test_curvature_extrinsic.py -vv --durations=10
```

The same `MICROSCOPE_TEST_DEVICE=cpu` override selects CPU for its analytic grid tests. Their fixed isometry seeds
make the calibrated decimal assertions reproducible; the integral uses fewer decimals to allow for coarse quadrature.

The analytic grids use float64, as they have historically; their dense finite differences need that precision.
Runtime and memory use depend on the selected device and method. Small tests with an explicit CPU device and tests
parameterized over CPU/CUDA retain their declared device coverage.

## Type checking

Pyright checks every maintained Python file, including tests, on every commit. The hook always runs the entire project
and does not limit analysis to staged files. Errors and warnings fail the hook directly; there is no accepted-error
baseline. Missing dependencies also fail, so activate the development environment before checking or committing.

Run the full check from the repository root:

```sh
python -m pyright --project pyproject.toml --warnings
```

Notebooks remain outside the command-line check. Diagnostic details can change with scientific dependency versions,
so use the same development environment when comparing results.

## VS Code diagnostics

Enable the Microsoft Python and Pylance extensions, then use **Python: Select Interpreter** to choose the development
environment. Open this repository as a workspace folder so Pylance reads `[tool.pyright]` from `pyproject.toml`.
For inline diagnostics and Problems entries throughout the workspace, use these VS Code settings:

```json
{
  "python.languageServer": "Pylance",
  "python.analysis.typeCheckingMode": "standard",
  "python.analysis.diagnosticMode": "workspace"
}
```

After changing the interpreter or configuration, run **Python: Restart Language Server**. Open **View: Toggle Problems**
from the Command Palette to inspect diagnostics. Both the editor and the pre-commit hook check tests as well as library
code.
Pylance bundles its own checker version, so editor diagnostics can differ slightly from the pinned command-line version.
See the [Python settings reference](https://code.visualstudio.com/docs/python/settings-reference).

## Style conventions

The hooks check code without rewriting it. Ruff's formatter is deliberately not a hook: its default collapsing of
short signatures/calls would undo the chosen multiline layouts. Keep signatures with three or more arguments on
separate lines, expand crowded calls, include abstract tensor/array shapes in argument docs, and put helpers before
callers. These layout, shape-description and ordering conventions still need review.

Ruff, Pyright and pre-commit versions are pinned in the `dev` extra in `pyproject.toml`; update that extra when upgrading
these tools. pre-commit 4.6.2 supports the project's Python 3.10 baseline. The Pyright `nodejs` extra supplies its Node.js
runtime within the Python environment without requiring a separate system installation. See the [Ruff configuration documentation](https://docs.astral.sh/ruff/configuration/),
[Pyright configuration documentation](https://github.com/microsoft/pyright/blob/main/docs/configuration.md) and
[pre-commit local-hook documentation](https://pre-commit.com/#repository-local-hooks) for the supported settings.
