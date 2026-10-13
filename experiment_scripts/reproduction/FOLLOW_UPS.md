# Follow-up status

The source fixes below are implemented on the cleanup branch. **They have not
been executed or tested.** Historical defaults are retained unless an option
explicitly changes them.

| Area | Implemented |
|---|---|
| Experiment paths | Inference/analysis discover actual six-field directories, including repetition, rather than building an incompatible grid. The second analysis entry point reuses the first. |
| Training selection | Dataset/model/dimension/fraction/repetition lists, explicit per-repetition seeds, workers, budget flags and JSON model overrides are exposed. The default shared seed remains available. |
| Checkpoints and stopping | Full trusted local solver checkpoints map to the selected device with explicit `weights_only=False`. Opt-in strict budgets stop at batch/epoch limits; historical epoch-finish behavior remains the default. |
| Geometry devices | Analysis forwards its device through `run_single_output` into grid measures. The pre-existing reach/derivative device fixes are preserved. |
| Fixed splits | FPS receives the selected device, supports empty selections and excludes duplicate indices. Conflicting cached effective test sizes raise without overwriting. |
| Toy fitting | Device reaches MMLS and surface CVT; analytic curves accept the shared option. Denoising/Lightning imports occur only when that method is selected. |
| Layer inference | Conditional pin memory, optional seed/model architecture and sampled/mean decoding. Native inference records these choices; sampled unseeded decoding remains the default. |
| Notebook | The methods notebook delegates evaluation/export/plotting to the new modules, with a configurable results root. Obsolete imports, absolute paths, duplicated evaluation and stale saved outputs were removed. |
| Completion | Older training/inference/analysis entry points now use the same `COMPLETED` semantics. Partial warnings print even if imported code filters Python warnings. No automated deletion/recovery. |

Focused regression sources were added for FPS reference/duplicate/empty cases,
cached-split preservation/device forwarding, strict budgets and historical
epoch finish, full-checkpoint NumPy metadata, mean decoding and partial warnings.
No tests, imports, CLI help or experiments have been run for these changes.

## Still requires validation or historical evidence

- Run the focused regressions, CLI option checks and small CPU/GPU pipelines.
  Compare fixed splits, checkpoint errors, bound curves and layer aggregates
  with retained artifacts before claiming reproduction.
- The cleanup branch fixes signed curvature integration, second-kind Riemann
  differentiation and subsampled reach shapes. Their numerical effect on the
  older paper outputs has not been measured. MPS remains unvalidated.
- Static source comparison with OLD confirms toy-schema renames:
  `MeasuresGlobal.max_principal` became `max_absolute_principal`, and
  `MeasuresLocal.principal` became `max_absolute_principal`. Reach field names
  remain the same; pickle module paths changed from geometry to microscope.
  Semantic/value agreement and migration of legacy pickles/YAML remain separate
  work; these scripts use native new-run artifacts.
- Historical evidence supports three dSprites repetitions, two retained COIL
  MMLS repetitions and twenty toy examples per size. Figure 5's generic caption
  says three. The recovered recipe follows retained evidence; recovering a
  missing COIL repetition/toy raw result or correcting the caption requires
  further evidence or an author decision.
- Bound constants follow the implemented notebook: log-log regression of mean
  errors, then 0.01/0.99 quantiles of fitted-curve/theoretical-shape ratios.
  The paper describes per-size repetition quantiles and envelope extrema.
  The original formula is preserved; a prose correction or alternative
  computation requires a separate decision/comparison.
- The reason for the historical dSprites inference fix is still unidentified;
  its inference RNG state is unrecorded. Exposed seed/mean-decoding options do
  not recover those unknowns.
- Handle partial directories manually or choose new output roots. Do not add
  completion markers merely because some checkpoints or plots exist. Markers
  do not compare settings; changes require fresh output directories.
