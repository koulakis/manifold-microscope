# Reproducing the paper experiments

Install the experiment dependencies with `python -m pip install '.[benchmark]'`.
Custom dSprites is generated locally. For COIL-20, set `COIL20_PATH` to the
directory containing its 20 × 72 processed grayscale PNGs, including `obj1__0.png`.
From the repository root, expand a figure configuration before launching it:

```bash
python -m experiments_and_analysis.scripts.sweep \
  --config experiments_and_analysis/scripts/configs/paper/figure4_quick.yaml \
  --output-dir ../outputs/new-paper-run --dry-run
```

Remove `--dry-run` to fit, infer, analyze and render the figure. Configurations for
Figures 3–6 are together in `experiments_and_analysis/scripts/configs/paper`.
The quick Figure 4 preset uses six sizes and one repetition; the full preset
uses thirteen sizes and three repetitions. Full configurations can take many
hours and require substantial memory; geometry uses the full structured grid.

Use one new working directory for related figures to reuse compatible fits.
The new pipeline does not import old reproduction directories. Historical
normalization, bound formulas and signed-curvature aggregation are retained.
Seeded execution does not establish exact equivalence to the original paper's
partly unseeded runs; small plot variation is expected.

See the [experiment guide](../experiments_and_analysis/README.md) for Python
calls, configuration fields, stage outputs, reuse and small custom workflows.
