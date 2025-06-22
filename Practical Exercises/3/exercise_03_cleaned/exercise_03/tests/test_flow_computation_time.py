import numpy as np
from .base_tests import UnitTest
from pathlib import Path
from PIL import Image
from exercise_code.getFlow import compute_flow
from time import time
import logging


class FlowTime(UnitTest):
    def __init__(self, dataset_dir,  seq_name='RubberWhale', tracker_name="Lukas Kanade", key="time to compute the optical flow"):
        self.tracker_name = tracker_name
        self.seq_name = seq_name
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
        # measure the time it takes to compute the optical flow
        t1 = time()  # start timer

        # Compute the optical flow using the Lucas-Kanade method
        flow_lk = compute_flow(im1_gray, im2_gray)
        t2 = time()

        self.time = t2 - t1
        # time in seconds

        return self.time
        
    def define_message(self):
        return f"Your {self.tracker_name} took {self.time:.2f} s {self.key} on sequence {self.seq_name}.\nTest passed"

def test_flow_computation_time(dataset_dir):
    test = FlowTime(dataset_dir)
    score = test.test()
    logging.info(test.define_message())
    return score
