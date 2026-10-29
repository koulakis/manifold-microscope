"""Utility helpers for the adapted beta-VAE implementation.

Copied and adapted from https://github.com/1Konny/Beta-VAE. The upstream project is MIT licensed; see this package's
`LICENSE` and the repository's `docs/source/third_party/beta_vae.rst` for copyright and attribution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


import argparse
import subprocess

if TYPE_CHECKING:
    import torch


def cuda(tensor: torch.Tensor, uses_cuda: bool) -> torch.Tensor:
    """Backward compatibility function for CUDA device placement.

    Args:
        tensor: Tensor of shape (...).
        uses_cuda: Move the tensor to CUDA when true.

    Returns:
        Tensor with unchanged shape (...), moved to CUDA when uses_cuda=True.
    """
    return tensor.cuda() if uses_cuda else tensor


def to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Move tensor to specified device (cpu, cuda, mps, etc.)

    Args:
        tensor: Tensor of shape (...).
        device: Device used for tensor computations.

    Returns:
        Tensor with unchanged shape (...) on the requested device.
    """
    return tensor.to(device)


def str2bool(v: str) -> bool:
    """Parse a command-line boolean string.

    Args:
        v: Boolean text such as yes/no, true/false or 1/0.

    Returns:
        True for a recognized true string and False for a recognized false string.
    """
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def where(
    cond: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """Do same operation as np.where

    code from:
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8

    Args:
        cond: Condition mask broadcastable to x and y, of shape (...).
        x: Values where the mask is true, of shape (...).
        y: Values where the mask is false, of shape (...).

    Returns:
        Tensor of the broadcast shape of cond, x and y, combining x and y according to cond.
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


def grid2gif(
    image_str: str,
    output_gif: str,
    delay: int = 100
) -> None:
    """Make GIF from images.

    code from:
        https://stackoverflow.com/a/34555939

    Args:
        image_str: ImageMagick input image expression.
        output_gif: Output GIF filename.
        delay: Frame delay in hundredths of a second.

    Returns:
        None.
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)
