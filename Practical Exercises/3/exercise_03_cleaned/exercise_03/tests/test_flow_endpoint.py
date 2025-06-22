import numpy as np
from .base_tests import UnitTest
from pathlib import Path
import cv2
from PIL import Image
from exercise_code.getFlow import compute_flow
import logging


class FlowEndpointError(UnitTest):
    def __init__(self, dataset_dir, seq_name='RubberWhale', tracker_name="Lukas Kanade", key="endpoint error"):
        self.seq_name = seq_name
        self.tracker_name = tracker_name
        self.dataset_dir = Path(dataset_dir)
        self.imgs = ['frame10.png', 'frame11.png']
        self.flow = ['flow10.flo']
        self.key = key
        self.results_name = "exercise_code/test/"+self.tracker_name+".pth"

    def test(self):
        im1 = Image.open(self.dataset_dir / "other-data" / self.seq_name / self.imgs[0])
        im2 = Image.open(self.dataset_dir / "other-data" / self.seq_name / self.imgs[1])
        im1_gray = np.array(im1.convert('L'))
        im2_gray = np.array(im2.convert('L'))

        # Read ground truth flow
        flow_path = self.dataset_dir / "other-gt-flow" / self.seq_name / self.flow[0]
        flow = cv2.readOpticalFlow(str(flow_path))

        # Compute the optical flow using the Lucas-Kanade method
        flow_lk = compute_flow(im1_gray, im2_gray)

        # compute the cosine similarity between the ground truth flow and the computed flow
        mask = np.linalg.norm(flow, axis=-1) < 2357022699
        endpoint_err = np.linalg.norm(flow - flow_lk, axis=-1)

        self.endpoint_err_mean = endpoint_err[mask].mean()
        return self.endpoint_err_mean
        
    def define_message(self):
        return f"Your {self.tracker_name} reached the mean {self.key} {self.endpoint_err_mean:.2f} on sequence {self.seq_name}.\nTest passed"


def test_flow_endpoint(dataset_dir):
    test = FlowEndpointError(dataset_dir)
    score = test.test()
    logging.info(test.define_message())
    return score
