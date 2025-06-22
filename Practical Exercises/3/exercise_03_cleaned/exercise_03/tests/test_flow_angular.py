import numpy as np
from .base_tests import UnitTest
from pathlib import Path
import cv2
from PIL import Image
from exercise_code.getFlow import compute_flow
import logging


class FlowAngularError(UnitTest):
    def __init__(self, dataset_dir, seq_name='RubberWhale', tracker_name="Lukas Kanade", key="cosine similarity (angular error)"):
        self.seq_name = seq_name
        self.tracker_name = tracker_name
        self.imgs = ['frame10.png', 'frame11.png']
        self.flow = ['flow10.flo']
        self.dataset_dir = Path(dataset_dir)
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
        flow_normalized = flow / np.linalg.norm(flow, axis=-1)[:, :, np.newaxis]

        # Compute the optical flow using the Lucas-Kanade method
        flow_lk = compute_flow(im1_gray, im2_gray)
        mask_lk = np.linalg.norm(flow_lk, axis=-1)<1e-5
        flow_lk_normalized= flow_lk/ np.linalg.norm(flow_lk, axis=-1)[:, :, np.newaxis]
        flow_lk_normalized[mask_lk] = 0

        # compute the cosine similarity between the ground truth flow and the computed flow
        mask = np.linalg.norm(flow, axis=-1) < 2357022699
        cos_sim = (flow_normalized * flow_lk_normalized).sum(axis=-1)

        self.ang_err = cos_sim[mask].mean()

        return self.ang_err
        
    def define_message(self):
        return f"Your {self.tracker_name} reached the mean {self.key} {self.ang_err:.2f} on sequence {self.seq_name}.\nTest passed"

def test_flow_angular(dataset_dir):
    test = FlowAngularError(dataset_dir)
    score = test.test()
    logging.info(test.define_message())
    return score
