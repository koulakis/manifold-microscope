"""Framework-independent boundary for caller-supplied trainable models."""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np


class TrainingPaused(RuntimeError):
    """Training saved a resumable checkpoint before finishing its configured budget."""


class ModelAdapter(Protocol):
    """A native model and its checkpoint/preprocessing state, owned by the caller."""

    @property
    def identifier(self) -> str:
        """Return the implementation/version recorded in model.adapter."""
        ...

    def fit(
        self,
        data: np.ndarray,
        output_dir: Path,
        seed: int,
        device: str
    ) -> list[str]:
        """Fit only supplied samples and save native artifacts.

        Args:
            data: Training pool of shape (n, *input_shape); validation must stay within it.
            output_dir: Stage directory, which already contains the pipeline manifest.
            seed: Training seed.
            device: Requested execution device.

        Returns:
            Relative filenames required to restore the selected model.
        """
        ...

    def load(self, output_dir: Path, device: str) -> None:
        """Restore selected weights and preprocessing from a completed fit.

        Args:
            output_dir: Native artifact directory.
            device: Requested execution device.
        """
        ...

    def representation_names(self) -> list[str]:
        """Return ordered input, encoder, bottleneck, decoder and output names."""
        ...

    def batches(
        self,
        data: np.ndarray,
        representation: str,
        batch_size: int,
        seed: int
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield deterministic, consecutive representations in input order.

        Args:
            data: Query array of shape (n, *input_shape).
            representation: One name from representation_names().
            batch_size: Maximum number of query samples per batch.
            seed: Inference seed.

        Yields:
            (start, values), with floating NumPy values of shape (b, *feature_shape).
        """
        ...
