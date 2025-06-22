import numpy as np
from .base_tests import UnitTest, CompositeTest, test_results_to_score
from exercise_code.utils import getM


class MZeroGradient(UnitTest):
    def __init__(self) -> None:
        # Create zero gradients: Ix = 0, Iy = 0
        H, W = 20, 20
        self.Ix = np.zeros((H, W), dtype=np.float32)
        self.Iy = np.zeros((H, W), dtype=np.float32)
        self.sigma = 1.0

    def test(self):
        M = getM(self.Ix, self.Iy, sigma=self.sigma)
        # M should be zero everywhere
        if not np.allclose(M, 0.0, atol=1e-6):
            diff = M[np.abs(M) > 1e-6]
            self.output = diff.flat[0] if diff.size > 0 else None
            return False
        return True

    def define_success_message(self):
        return "MZeroGradient passed: zero gradients yield zero structure tensor everywhere."

    def define_failure_message(self):
        return "MZeroGradient failed: expected M all zeros, but found nonzero entry {}.".format(
            self.output
        )
    
    
class MInteriorTest(UnitTest):
    def __init__(self) -> None:
        self.sigma = 0.5

        self.Ix = np.asarray([
            [1, 1, 1, 1],
            [1, 1, 2, 1],
            [1, 2, 1, 1],
            [1, 1, 1, 1]
        ], dtype=np.float32)

        self.Iy = np.ones((4, 4), dtype=np.float32)

        # Expected M values at interior points (manually copied from reference)
        self.expected = {
            (1, 1): np.array([[1.50291703, 1.16763901], [1.16763901, 1.0]]),
            (1, 2): np.array([[2.8920723,  1.63069077], [1.63069077, 1.0]]),
            (2, 1): np.array([[2.8920723,  1.63069077], [1.63069077, 1.0]]),
            (2, 2): np.array([[1.50291703, 1.16763901], [1.16763901, 1.0]])
        }

    def test(self):
        M = getM(self.Ix, self.Iy, sigma=self.sigma)

        for (i, j), expected_ij in self.expected.items():
            actual_ij = M[i, j]
            if not np.allclose(actual_ij, expected_ij, atol=1e-6):
                self.output = (i, j, actual_ij, expected_ij)
                return False

        return True

    def define_success_message(self):
        return "MInteriorHardcodedValuesTest passed: interior M values match expected results."

    def define_failure_message(self):
        i, j, actual, expected = self.output
        return (
            f"M value mismatch at pixel ({i}, {j}):\n"
            f"Got:\n{actual}\nExpected:\n{expected}"
        )

    
class MShapeTest(UnitTest):
    def __init__(self) -> None:
        # create random gradient image with known shape
        self.H, self.W = 12, 17
        self.Ix = np.random.randn(self.H, self.W).astype(np.float32)
        self.Iy = np.random.randn(self.H, self.W).astype(np.float32)
        self.sigma = 1.0

    def test(self):
        M = getM(self.Ix, self.Iy, sigma=self.sigma)
        # check that M has shape (H, W, 2, 2)
        expected_shape = (self.H, self.W, 2, 2)
        if M.shape != expected_shape:
            self.output = (M.shape, expected_shape)
            return False
        return True

    def define_success_message(self):
        return f"MShapeTest passed: M has correct shape."

    def define_failure_message(self):
        found, expected = self.output
        return f"MShapeTest failed: got shape {found}, expected {expected}."


class MTests(CompositeTest):
    def define_tests(self):
        return [
            MShapeTest(),
            MZeroGradient(),
            MInteriorTest()
        ]


def test_getM():
    test = MTests()
    return test_results_to_score(test())
