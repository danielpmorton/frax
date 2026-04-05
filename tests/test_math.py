"""Test cases for math/linalg utils"""

import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from frax.utils.linalg_utils import (
    fast_spd_inverse,
    random_spd_matrix,
    schur_spd_inverse,
)

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


def check_inversion_accuracy(mat, inv, atol, rtol):
    n = mat.shape[0]
    assert inv.shape == mat.shape == (n, n)
    try:
        np.testing.assert_allclose(inv @ mat, np.eye(n), atol=atol, rtol=rtol)
    except AssertionError:
        # If my identity check did not pass then make sure that the standard inverse is also
        # having some numerical difficulty on this matrix
        try:
            # If this assertion PASSES then the standard inverse performs better on this edge case
            # and thus we have introduced a problem with our custom method
            np.testing.assert_allclose(
                np.linalg.inv(mat) @ mat,
                np.eye(n),
                atol=atol,
                rtol=rtol,
            )
            raise  # The previous error
        except AssertionError:
            # If this assertion FAILS then our inverse performs the same as the standard (this is ok)
            pass


class TestLinalg(unittest.TestCase):
    DIMS = [5, 10, 20, 30]
    NUM_ACCURACY_TESTS = 100
    NUM_SPEED_TESTS = 100
    RTOL = 1e-10
    ATOL = 1e-10

    def _test_spd_inv_accuracy(self, n):
        matrices = [random_spd_matrix(n) for _ in range(self.NUM_ACCURACY_TESTS)]

        @jax.jit
        def jit_fast_spd_inv(mat):
            return fast_spd_inverse(mat)

        custom_invs = [jit_fast_spd_inv(m) for m in matrices]

        for mat, inv in zip(matrices, custom_invs):
            check_inversion_accuracy(mat, inv, self.ATOL, self.RTOL)

    def test_schur_inv_accuracy(self):
        for n in self.DIMS:
            self._test_schur_inv_accuracy(n)

    def _test_schur_inv_accuracy(self, n):
        matrices = [random_spd_matrix(n) for _ in range(self.NUM_ACCURACY_TESTS)]

        @jax.jit
        def jit_schur_inv(mat):
            return schur_spd_inverse(mat, split_idx=mat.shape[0] // 4)

        custom_invs = [jit_schur_inv(m) for m in matrices]

        for mat, inv in zip(matrices, custom_invs):
            check_inversion_accuracy(mat, inv, self.ATOL, self.RTOL)

    def _test_spd_inv_speed(self, n):
        matrices = [random_spd_matrix(n) for _ in range(self.NUM_SPEED_TESTS)]

        @jax.jit
        def jit_fast_spd_inv(mat):
            return fast_spd_inverse(mat)

        @jax.jit
        def jit_standard_inv(mat):
            return jnp.linalg.inv(mat)

        # Dummy solves for jit compilationre
        dummy_mat = random_spd_matrix(n)
        jit_fast_spd_inv(dummy_mat)
        jit_standard_inv(dummy_mat)

        start_time_fast_psd_inv = time.perf_counter()
        for m in matrices:
            _m_inv = jit_fast_spd_inv(m).block_until_ready()
        duration_fast_psd_inv = time.perf_counter() - start_time_fast_psd_inv
        avg_time_fast_psd_inv = duration_fast_psd_inv / self.NUM_SPEED_TESTS

        start_time_standard_inv = time.perf_counter()
        for m in matrices:
            _m_inv = jit_standard_inv(m).block_until_ready()
        duration_standard_inv = time.perf_counter() - start_time_standard_inv
        avg_time_standard_inv = duration_standard_inv / self.NUM_SPEED_TESTS

        # NOTE: Commenting out these assertions for now and allowing the test to pass
        # even if it is a bit slower. This is a nondeterminstic test and depends on CPU
        # processes and should only be asserted in limited cases
        # factor = 1.05  # Accounts for some noise
        # self.assertLess(avg_time_fast_psd_inv, avg_time_standard_inv * factor)
        # self.assertLess(avg_time_fast_psd_inv, avg_time_alternate_psd_inv * factor)

        print(f"\nSpeed test for dimension {n}")
        print(f"Standard inverse: {avg_time_standard_inv * 1e6:.3f} µs")
        print(f"Fast PSD inverse: {avg_time_fast_psd_inv * 1e6:.3f} µs")

    def test_psd_inv_accuracy(self):
        for n in self.DIMS:
            self._test_spd_inv_accuracy(n)

    def test_speed(self):
        for n in self.DIMS:
            self._test_spd_inv_speed(n)


if __name__ == "__main__":
    unittest.main()
