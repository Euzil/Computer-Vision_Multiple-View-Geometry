import numpy as np
from exercise_code.utils import getM, getGradients, getTemporalPartialDerivative, getq

def getFlow(M, q, points):
    # Calculate the flow between two images
    # Input:
    # M: numpy.ndarray (structure tensor) of shape (H, W, 2, 2)
    # q: numpy.ndarray (q vector) of shape (H, W, 2)
    # points: numpy.ndarray (points) of shape (N, 2)
    # Output:
    # v: numpy.ndarray (flow) of shape (H, W, 2)
    # v_points: numpy.ndarray (flow for the points) of shape (N, 2)

    ########################################################################
    # TODO:                                                                #
    # Calculate the flow for each point by solving the linear system.      #
    #                                                                      #
    ########################################################################

    H, W = M.shape[:2]
    v = np.zeros((H, W, 2))
    
    # Solve M * v = -q for each pixel
    for i in range(H):
        for j in range(W):
            M_pixel = M[i, j]  # 2x2 matrix
            q_pixel = q[i, j]  # 2x1 vector
            
            # Check if matrix is invertible (determinant != 0)
            det = np.linalg.det(M_pixel)
            
            if abs(det) > 1e-10:  # Avoid division by zero
                # Solve: M * v = -q  =>  v = M^(-1) * (-q)
                v[i, j] = np.linalg.solve(M_pixel, -q_pixel)
            else:
                # If matrix is singular, set flow to zero
                v[i, j] = np.array([0.0, 0.0])
    
    # Extract flow vectors at specific points
    v_points = np.zeros((len(points), 2))
    for idx, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        # Make sure coordinates are within bounds
        if 0 <= x < W and 0 <= y < H:
            v_points[idx] = v[y, x]  # Note: v is indexed as [y, x]
        else:
            v_points[idx] = np.array([0.0, 0.0])

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return v, v_points


def compute_flow(im1_gray, im2_gray):
    # Compute the optical flow using the Lucas-Kanade method.
    # Input:
    # im1_gray: numpy.ndarray (first image) of shape (H, W)
    # im2_gray: numpy.ndarray (second image) of shape (H, W)
    # Output:
    # flow_lk: numpy.ndarray (flow) of shape (H, W, 2)

    use_cv2=False
    # The following code is used to compute the optical flow using the Lucas-Kanade method which is implemented in the getFlow function.
    if use_cv2:
        import cv2
        # get dense points for flow
        x = np.arange(0, im1_gray.shape[0], 1)
        y = np.arange(0, im1_gray.shape[1], 1)
        x, y = np.meshgrid(x, y)
        p0 = np.array([y.flatten(), x.flatten()]).T # opencv uses different order
        p0 = p0[:,None,:].astype(np.float32)
        lk_params = dict( winSize = (15, 15),
        maxLevel = 0,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        p1, st, err = cv2.calcOpticalFlowPyrLK(im1_gray, im2_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        flow_lk = np.zeros((im1_gray.shape[0], im1_gray.shape[1], 2))
        # fill in the flow
        for i in range(len(good_new)):
            try:
                flow_lk[int(good_old[i][1]), int(good_old[i][0])] = good_new[i] - good_old[i] # opencv uses different order
            except:
                pass
    else:
        x = np.arange(0, im1_gray.shape[0], 1)
        y = np.arange(0, im1_gray.shape[1], 1)
        x, y = np.meshgrid(x, y)
        points = np.array([x.flatten(), y.flatten()]).T
        It = getTemporalPartialDerivative(im1_gray, im2_gray)
        Ix, Iy = getGradients(im1_gray)
        q = getq(It, Ix, Iy)
        M = getM(Ix, Iy)# get dense points for flow
        flow_lk, flow_lk_points = getFlow(M, q, points)
        
    return flow_lk