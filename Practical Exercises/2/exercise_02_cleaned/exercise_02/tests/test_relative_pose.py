import numpy as np
from .base_tests import UnitTest, test_results_to_score

from exercise_code.camera import *

class RelativePose(UnitTest):
    def __init__(self) -> None:
        data = np.load("data/data.npz")
        self.relative_pose = data["relative_pose"]
        self.pose_1 = data["pose_1"]
        self.pose_2 = data["pose_2"]
        self.output = None
    def test(self):
        relative_pose = compute_relative_pose(self.pose_1, self.pose_2)
        self.output = relative_pose
        return np.allclose(relative_pose, self.relative_pose)

    def define_success_message(self):
        return f"Congratulations: You passed the test case for the relative pose."

    def define_failure_message(self):
        return f"The output of the relative pose is incorrect (expected {self.ref_pix}, got {self.output})."



def test_relative_pose():
    test = RelativePose()
    return test_results_to_score(test())
