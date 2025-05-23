import numpy as np
from .base_tests import UnitTest, CompositeTest, test_results_to_score

from exercise_code.camera import *

class Reprojection(UnitTest):
    def __init__(self,camType) -> None:
        self.camType = camType
        data = np.load("data/data.npz")
        self.pix = data["pixels"]
        self.distances = data["distances"]
        self.relative_pose = compute_relative_pose(data["pose_1"], data["pose_2"])
        self.ref_pix = data["reprojected_pixels_1"] if camType=="pinhole" else data["reprojected_pixels_2"]
        self.output = None
    def test(self):
        if self.camType == "pinhole":
            cam = Pinhole(640, 480, 600, 600, 320, 240)
        else:
            cam = Fov(640, 480, 600, 600, 320, 240, 0.1)
        npoints = self.ref_pix.shape[0]
        reprojected_pix = np.zeros_like(self.ref_pix)
        for i in range(npoints):
            pt = cam.unproject(self.pix[i,:], self.distances[i])
            pt_cam2 = self.relative_pose @ np.append(pt, 1.0)
            reprojected_pix[i,:] = cam.project(pt_cam2)
        self.output = reprojected_pix
        return np.allclose(reprojected_pix, self.ref_pix)

    def define_success_message(self):
        return f"Congratulations: You passed the test case for the {self.camType} reprojection."

    def define_failure_message(self):
        return f"The output of the {self.camType} reprojection is incorrect (expected {self.ref_pix}, got {self.output})."



class ReprojectionTest(CompositeTest):
    def define_tests(self):
        return [
            Reprojection("pinhole"),
            Reprojection("FOV")
        ]

def test_reprojection():
    test = ReprojectionTest()
    return test_results_to_score(test())
