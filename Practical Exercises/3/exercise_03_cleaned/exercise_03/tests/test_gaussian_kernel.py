import numpy as np
from .base_tests import UnitTest, CompositeTest, test_results_to_score
from exercise_code.utils import getGaussiankernel


class GaussianKernelSum(UnitTest):
    def __init__(self) -> None:
        # Test different sigma values
        self.sigmas = [2, 5, 7]

    def test(self):
        for sigma in self.sigmas:
            G = getGaussiankernel(sigma)
            # discrete Gaussian kernel should sum to 1 
            if not np.allclose(np.sum(G), 1.0, atol=1e-6):
                self.output = (sigma, np.sum(G))
                return False
        return True

    def define_success_message(self):
        return "GaussianKernelSum passed: all kernels sum to 1."

    def define_failure_message(self):
        sigma, total = self.output
        return f"GaussianKernelSum failed: kernel sum is not 1 for σ={sigma} (sum={total})."


class GaussianKernelSymmetry(UnitTest):
    def __init__(self) -> None:
        self.sigmas = [2, 5, 7]

    def test(self):
        for sigma in self.sigmas:
            G = getGaussiankernel(sigma)
            H, W = G.shape
            for i in range(H):
                for j in range(W):
                    if not np.allclose(G[i, j], G[H - 1 - i, W - 1 - j], atol=1e-6):
                        self.output = (sigma, i, j, float(G[i, j]), float(G[H - 1 - i, W - 1 - j]))
                        return False
        return True

    def define_success_message(self):
        return "GaussianKernelSymmetry passed: all kernels are symmetric."

    def define_failure_message(self):
        sigma, i, j, found, expected = self.output
        H, W = getGaussiankernel(sigma).shape
        return (
            f"GaussianKernelSymmetry failed for σ={sigma}: "
            f"G[{i},{j}] = {found:.6f} but G[{H-1-i},{W-1-j}] = {expected:.6f}."
        )


class GaussianKernelHardcoded(UnitTest):
    def __init__(self) -> None:
        # expected kernels
        self.expected = {
            0.5: np.array([
                [0.011344, 0.083820, 0.011344],
                [0.083820, 0.619347, 0.083820],
                [0.011344, 0.083820, 0.011344]
            ], dtype=np.float32),
            1.0: np.array([
                [0.002969, 0.013306, 0.021938, 0.013306, 0.002969],
                [0.013306, 0.059634, 0.098320, 0.059634, 0.013306],
                [0.021938, 0.098320, 0.162103, 0.098320, 0.021938],
                [0.013306, 0.059634, 0.098320, 0.059634, 0.013306],
                [0.002969, 0.013306, 0.021938, 0.013306, 0.002969]
            ], dtype=np.float32)
        }
        self.sigmas = [0.5, 1.0]

    def test(self):
        for sigma in self.sigmas:
            G = getGaussiankernel(sigma)
            expected = self.expected[sigma]
            if G.shape != expected.shape or not np.allclose(G, expected, atol=1e-3):
                self.output = sigma
                return False
        return True

    def define_success_message(self):
        return "GaussianKernelHardcoded passed: exact values correct for σ=0.5 and σ=1.0."

    def define_failure_message(self):
        sigma = self.output
        return f"GaussianKernelHardcoded failed: kernel for σ={sigma} does not match expected values."


class GaussianKernelTests(CompositeTest):
    def define_tests(self):
        return [
            GaussianKernelSum(),
            GaussianKernelSymmetry(),
            GaussianKernelHardcoded(),
        ]


def test_getGaussiankernel():
    test = GaussianKernelTests()
    return test_results_to_score(test())
