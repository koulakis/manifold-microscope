import unittest

import numpy as np

from microscope.patches import extract_patches, stack_patches


class TestPatches(unittest.TestCase):
    def test_slicing_to_patches_and_reconstructing_array(self):
        array = np.random.rand(10, 12, 8, 3, 7)
        patch_sizes = [4, 3, 6]
        overlaps = [2, 0, 4]
        patches = extract_patches(array, patch_sizes, overlaps)

        reconstructed_array = stack_patches(
            patches[:, :, :, 1:-1, :, 2:-2, ...],
            n_features=2)

        np.testing.assert_array_equal(array[1:-1, :, 2:-2, ...], reconstructed_array)
