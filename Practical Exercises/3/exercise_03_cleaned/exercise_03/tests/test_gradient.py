import numpy as np
from .base_tests import UnitTest, CompositeTest, test_results_to_score

# Import the function to be tested
from exercise_code.utils import getGradients


class GradientZeroImage(UnitTest):
    def __init__(self) -> None:
        # constant image (all pixels the same)
        self.I = np.zeros((10, 10), dtype=np.float32)

    def test(self):
        # gradients with no smoothing (sigma=0)
        Ix, Iy = getGradients(self.I, sigma=0)
        # gradients should be (approximately) zero everywhere
        return np.allclose(Ix, 0.0) and np.allclose(Iy, 0.0)

    def define_success_message(self):
        return "GradientZeroImage passed: Zero input yields zero gradients."

    def define_failure_message(self):
        return "GradientZeroImage failed: Expected zero gradients for a constant image."


class GradientHorizontalRamp(UnitTest):
    def __init__(self) -> None:
        # Horizontal ramp image: I[i, j] = j
        W, H = 5, 5
        self.I = np.tile(np.arange(W, dtype=np.float32), (H, 1))
        self.expected_Ix = np.full((H, W), 1.0 / 255.0, dtype=np.float32)
        self.expected_Ix[:, 0] = self.expected_Ix[:, -1] = 1.0 / (2 * 255.0)
        self.expected_Iy = np.zeros((H, W), dtype=np.float32)

    def test(self):
        Ix, Iy = getGradients(self.I, sigma=0)

        if not np.allclose(Ix, self.expected_Ix, atol=1e-6):
            self.output = ("Ix", Ix, self.expected_Ix)
            return False
        if not np.allclose(Iy, self.expected_Iy, atol=1e-6):
            self.output = ("Iy", Iy, self.expected_Iy)
            return False

        return True

    def define_success_message(self):
        return "GradientHorizontalRamp passed: Horizontal ramp yields correct Ix and zero Iy."

    def define_failure_message(self):
        label, actual, expected = self.output
        return f"{label} mismatch: got {actual}, expected {expected}."


class GradientVerticalRamp(UnitTest):
    def __init__(self) -> None:
        # Vertical ramp image: I[i, j] = i
        W, H = 5, 5
        self.I = np.tile(np.arange(H, dtype=np.float32).reshape(H, 1), (1, W))
        self.expected_Iy = np.full((H, W), 1.0 / 255.0, dtype=np.float32)
        self.expected_Iy[0, :] = self.expected_Iy[-1, :] = 1.0 / (2 * 255.0)
        self.expected_Ix = np.zeros((H, W), dtype=np.float32)

    def test(self):
        Ix, Iy = getGradients(self.I, sigma=0)

        if not np.allclose(Iy, self.expected_Iy, atol=1e-6):
            self.output = ("Iy", Iy, self.expected_Iy)
            return False
        if not np.allclose(Ix, self.expected_Ix, atol=1e-6):
            self.output = ("Ix", Ix, self.expected_Ix)
            return False

        return True

    def define_success_message(self):
        return "GradientVerticalRamp passed: Vertical ramp yields correct Iy and zero Ix."

    def define_failure_message(self):
        label, actual, expected = self.output
        return f"{label} mismatch: got {actual}, expected {expected}."


class GradientTests(CompositeTest):
    def define_tests(self):
        return [
            GradientZeroImage(),
            GradientHorizontalRamp(),
            GradientVerticalRamp(),
        ]


def test_gradients():
    test = GradientTests()
    return test_results_to_score(test())
