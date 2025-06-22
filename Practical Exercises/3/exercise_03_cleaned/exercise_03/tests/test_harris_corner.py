import numpy as np
from .base_tests import UnitTest, CompositeTest, test_results_to_score
from exercise_code import getHarrisCorners


class HarrisCornerCenter(UnitTest):
    def __init__(self) -> None:
        # 3×3 structure tensor stack M
        # Only the center pixel strong corner response.
        H, W = 3, 3
        self.M = np.zeros((H, W, 2, 2), dtype=np.float32)
        # At center (1,1), set M = identity 
        self.M[1, 1, 0, 0] = 1.0  # M11
        self.M[1, 1, 1, 1] = 1.0  # M22
        
        self.expected_center_score = 0.8
        # Off-diagonals are zero by default

        # Harris parameters
        self.kappa = 0.05
        self.theta = 1e-6

    def test(self):
        score, points = getHarrisCorners(self.M, kappa=self.kappa, theta=self.theta)
  
        if not np.allclose(score[1, 1], self.expected_center_score, atol=1e-6):
            self.output = ("score_center", float(score[1, 1]), self.expected_center_score)
            return False
        # check no other pixel has a score above zero (within tolerance)
        score_copy = score.copy()
        score_copy[1, 1] = 0.0
        if not np.allclose(score_copy, 0.0, atol=1e-6):
            # find a pixel that incorrectly has non-zero score
            nonzero_idx = np.argwhere(np.abs(score_copy) > 1e-6)[0]
            self.output = ("extra_score", tuple(nonzero_idx), float(score_copy[tuple(nonzero_idx)]))
            return False

        # check corners: (1,1) should be the only corner
        if points.shape != (1, 2):
            self.output = ("count", points.shape[0], 1)
            return False

        if not np.array_equal(points[0], np.array([1, 1])):
            self.output = ("location", tuple(points[0]), (1, 1))
            return False

        return True

    def define_success_message(self):
        return "HarrisCornerCenter passed: detected exactly the center corner with correct score."

    def define_failure_message(self):
        kind, found, expected = self.output
        if kind == "score_center":
            return f"HarrisCornerCenter failed: center score {found:.6f} ≠ expected {expected:.6f}."
        if kind == "extra_score":
            y, x = found
            return f"HarrisCornerCenter failed: unexpected non-zero score {expected:.6f} at pixel ({y},{x})."
        if kind == "count":
            return f"HarrisCornerCenter failed: detected {found} corners, expected 1."
        # kind == "location"
        yx_found = found
        yx_expected = expected
        return f"HarrisCornerCenter failed: detected corner at {yx_found}, expected {yx_expected}."


class HarrisCornerEdgePoints(UnitTest):
    def __init__(self) -> None:
        # 5×5 structure tensor stack M
        H, W = 5, 5
        self.M = np.zeros((H, W, 2, 2), dtype=np.float32)
        # two corner responses at (0,0) and (4,4)
        self.M[0, 0, 0, 0] = 5.0
        self.M[0, 0, 1, 1] = 1.0
        
        self.M[4, 4, 0, 0] = 1.0
        self.M[4, 4, 1, 1] = 1.0

        self.kappa = 0.05
        self.theta = 1e-6
        
        self.expected_score = (3.2, 0.8)

    def test(self):
        score, points = getHarrisCorners(self.M, kappa=self.kappa, theta=self.theta)

        if not np.allclose(score[0, 0], self.expected_score[0], atol=1e-3):
            self.output = ("score_00", float(score[0, 0]), self.expected_score[0])
            return False
        if not np.allclose(score[4, 4], self.expected_score[1], atol=1e-3):
            self.output = ("score_44", float(score[4, 4]), self.expected_score[1])
            return False

        # other scores should be zero
        mask = np.ones_like(score, dtype=bool)
        mask[0, 0] = False
        mask[4, 4] = False
        if not np.allclose(score[mask], 0.0, atol=1e-6):
            idx = np.argwhere((np.abs(score) > 1e-6) & mask)[0]
            self.output = ("extra_score", tuple(idx), float(score[tuple(idx)]))
            return False

        # check detected corners
        expected_points = {(0, 0), (4, 4)}
        found_points = {tuple(pt) for pt in points}
        if found_points != expected_points:
            self.output = ("points", found_points, expected_points)
            return False

        return True

    def define_success_message(self):
        return "HarrisCornerEdgePoints passed: detected corners at (0,0) and (4,4) with correct scores."

    def define_failure_message(self):
        kind, *details = self.output
        if kind.startswith("score_"):
            pos = kind.split("_")[1]
            found, expected = details
            return f"HarrisCornerEdgePoints failed: score at {pos} = {found:.6f}, expected {expected:.6f}."
        if kind == "extra_score":
            (y, x), val = details
            return f"HarrisCornerEdgePoints failed: unexpected non-zero score {val:.6f} at pixel ({y},{x})."
        # kind == "points"
        found_set, expected_set = details
        return f"HarrisCornerEdgePoints failed: found points {found_set}, expected {expected_set}."


class HarrisCornerThreshold(UnitTest):
    def __init__(self) -> None:
        # Use the same 3×3 M as in the previous test
        H, W = 3, 3
        self.M = np.zeros((H, W, 2, 2), dtype=np.float32)
        self.M[1, 1, 0, 0] = 1.0
        self.M[1, 1, 1, 1] = 1.0

        # choose a threshold above the center score
        self.kappa = 0.05
        # choose theta = 0.9 so no corner passes.
        self.theta = 0.9

    def test(self):
        score, points = getHarrisCorners(self.M, kappa=self.kappa, theta=self.theta)
        # Center score is 0.8 < theta = 0.9, so no points should be returned
        if points.size != 0:
            self.output = ("count", points.shape[0])
            return False

        # Additionally verify that the returned score matrix still has correct center value
        expected_center_score = 1.0 - self.kappa * (2.0**2)  # 0.8
        if not np.allclose(score[1, 1], expected_center_score, atol=1e-6):
            self.output = ("score_center", float(score[1, 1]), expected_center_score)
            return False

        return True

    def define_success_message(self):
        return "HarrisCornerThreshold passed: no corners detected when threshold > center score."

    def define_failure_message(self):
        kind, value = self.output
        if kind == "count":
            return f"HarrisCornerThreshold failed: detected {value} corners, expected 0."
        # kind == "score_center"
        found, expected = value
        return f"HarrisCornerThreshold failed: center score {found:.6f} ≠ expected {expected:.6f}."
    

class HarrisCornerTests(CompositeTest):
    def define_tests(self):
        return [
            HarrisCornerCenter(),
            HarrisCornerThreshold(),
            HarrisCornerEdgePoints()
        ]


def test_getHarrisCorners():
    test = HarrisCornerTests()
    return test_results_to_score(test())
