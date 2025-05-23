import numpy as np
from .base_tests import UnitTest, CompositeTest, test_results_to_score

from exercise_code.camera import *

class FovProject(UnitTest):
    def __init__(self) -> None:
        data = np.load("data/data.npz")
        self.ref_pix = data["pixels"]
        self.point = data["points2project_2"]
        self.output = None
    def test(self):
        cam = Fov(640, 480, 600, 600, 320, 240, 0.1)
        npoints = self.ref_pix.shape[0]
        pix = np.zeros_like(self.ref_pix)
        for i in range(npoints):
            pix[i,:] = cam.project(self.point[i,:])
        self.output = pix
        return np.allclose(pix, self.ref_pix)

    def define_success_message(self):
        return f"Congratulations: You passed the test case for the FOV projection."

    def define_failure_message(self):
        return f"The output of the FOV projection is incorrect (expected {self.ref_pix}, got {self.output})."
    

class FovUnproject(UnitTest):
    def __init__(self) -> None:
        data = np.load("data/data.npz")
        self.ref_points = data["points2project_2"]
        self.distances = data["distances"]
        self.pix = data["pixels"]
        self.output = None
    def test(self):
        cam = Fov(640, 480, 600, 600, 320, 240, 0.1)
        npix = self.ref_points.shape[0]
        points = np.zeros_like(self.ref_points)
        for i in range(npix):
            points[i,:] = cam.unproject(self.pix[i,:], self.distances[i])
        self.output = points
        return np.allclose(points, self.ref_points)

    def define_success_message(self):
        return f"Congratulations: You passed the test case for the FOV unprojection."

    def define_failure_message(self):
        return f"The output of the FOV unprojection is incorrect (expected {self.ref_points}, got {self.output})."
    


class FovTest(CompositeTest):
    def define_tests(self):
        return [
            FovProject(),
            FovUnproject()
        ]


def test_fov():
    test = FovTest()
    return test_results_to_score(test())
