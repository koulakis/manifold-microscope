# beta-VAE

Adapted from 1Konny/Beta-VAE. Model architectures and training losses are retained.
See LICENSE and NOTICE.md for attribution.

Training accepts an explicit array through `fitting.fit`; inference exposes
named representations through `inference.batches`. Run experiments using
`python -m experiments_and_analysis.scripts.run --config ... --output-dir ...`.
