# Paper reproduction

Reproduce the experiments from [*The Data Manifold under the Microscope* (ICML 2026)](https://proceedings.mlr.press/v306/koulakis26a.html) with the commands below.

**Note:** Paper reproduction is not fully seeded, so plots may vary slightly between runs. They should retain similar shapes and values in comparable ranges.

## Setup

```bash
pip install "manifold-microscope[benchmark]"
```

For a source checkout, use `pip install -e '.[benchmark]'` instead.

Custom dSprites is generated locally. For COIL-20, download and extract the
processed images from [Columbia's dataset page](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php).
Set `COIL20_PATH` to the directory containing all 20 × 72 grayscale PNGs,
including `obj1__0.png`.

## Run the paper experiments

Run the full Figures 3–6 pipeline:

```bash
export COIL20_PATH=/path/to/coil-20-proc
python -m experiment_scripts.reproduction.paper \
    --output-root results/paper --figures 3 4 5 6 --device cuda:0
```

Select individual figures with `--figures 3` or `--figures 3 5`. Full VAE
training can take days; `--device cpu` is supported but slower. Figure files
are written to `results/paper/figures/figure_<number>/` as PNG and PDF.

Run stages separately against the same output root, for example:

```bash
python -m experiment_scripts.reproduction.paper \
    --output-root results/paper --figures 3 --stages fit
python -m experiment_scripts.reproduction.paper \
    --output-root results/paper --figures 3 --stages export plot
```

Completed stages skip on restart. Preserve or move incomplete outputs before
retrying, and use a fresh output root when changing settings. MMLS defaults to
`thin_svd`; add `--projection-method original` for the historical covariance SVD.

See the [experiment guide](../experiment_scripts/reproduction/README.md) for
single-model MMLS/beta-VAE fitting and analysis, toy experiments, reduced
Figure 4 runs, and Figure 6 recovery with one layer per process. Retain your
source version, environment and command alongside results for later comparisons.
