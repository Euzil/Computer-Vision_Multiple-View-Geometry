import numpy as np
import cv2

def drawPoints(img, pts):
    # Draw points on image

    # Input:
    # img: numpy.ndarray

    # Output:
    # img: numpy.ndarray
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(pts.shape[1]):
        pt = (int(pts[0,i]), int(pts[1,i]))
        cv2.circle(img, pt, 10, (0, 255, 0),-1)
    return img