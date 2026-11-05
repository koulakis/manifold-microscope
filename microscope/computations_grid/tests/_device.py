"""Explicit device selection for the dense numerical reference tests."""

import os

import torch


def get_test_device() -> str:
    """Select CUDA by default, requiring an explicit environment override for CPU.

    Returns:
        Device name from MICROSCOPE_TEST_DEVICE, defaulting to cuda:0.
    """
    device = torch.device(os.environ.get("MICROSCOPE_TEST_DEVICE", "cuda:0"))
    if device.type == "cpu":
        return str(device)
    if device.type != "cuda":
        raise ValueError("MICROSCOPE_TEST_DEVICE must select cpu or cuda[:index].")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the dense grid tests but is unavailable. "
            "Check the CUDA setup, or set MICROSCOPE_TEST_DEVICE=cpu to run on CPU explicitly."
        )
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"MICROSCOPE_TEST_DEVICE selects an unavailable CUDA device: {device}.")
    return str(device)
