import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm, logm


def skewMat(w):
    # Create a skew-symmetric matrix from a vector w.
    # The result should be a 3x3 numpy array.
    # Input: w - a numpy array of shape (3,)
    # Output: \hat{w} - a 3x3 numpy array

    ########################################################################
    # TODO: Implement the skew-symmetric matrix,  you only need to simply   #
    # write down the skew-symmetric matrix here.                           # 
    ########################################################################


    w_hat = np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return w_hat
            


def exponentialSkewMat(w):
    # Create a skew-symmetric matrix from a vector w
    # return the exponential of the skew-symmetric matrix which should be rotation matrix
    w_hat = skewMat(w)
    return expm(w_hat)


def generateRandomPts(n_pts, R, T):
    # Generate random 3D points and project them to two cameras
    # input: number of points, rotation matrix, translation vector
    # output: 2D points in two cameras, camera intrinsic matrix, 3D points

    # assume the camera intrinsic matrix is identity
    K1 = np.eye(3)
    K2 = np.eye(3)

    # array to store the projected 2D points, (x1, y1) and (x2, y2)
    x1 = np.zeros(n_pts)
    x2 = np.zeros(n_pts)
    y1 = np.zeros(n_pts)
    y2 = np.zeros(n_pts)

    # array to store the 3D points
    X1_gt = np.zeros((3, n_pts))

    # counter for the number of valid points
    n = 0
    max_attempts = n_pts * 100  # Prevent infinite loops
    attempts = 0
    
    while n < n_pts and attempts < max_attempts:
        attempts += 1
        
        # Generate random 3D point with better distribution
        # Strategy: generate points that are more likely to be visible in both cameras
        
        # Method 1: Generate points in a reasonable 3D space
        # X, Y coordinates: reasonable spread around origin
        # Z coordinate: ensure positive depth with good range
        x_coord = (np.random.rand() - 0.5) * 8  # [-4, 4]
        y_coord = (np.random.rand() - 0.5) * 8  # [-4, 4] 
        z_coord = np.random.rand() * 8 + 2      # [2, 10] - always positive, reasonable depth
        
        X1 = np.array([x_coord, y_coord, z_coord])

        # compute the 3D point in the second camera
        X2 = R @ X1 + T

        # check if the 3D points are in front of both cameras with sufficient depth
        if X1[2] > 0.5 and X2[2] > 0.5:  # Increased minimum depth for stability

            ########################################################################
            # TODO: project the 3D points to the two cameras                      #
            # Hint: use the camera intrinsic matrix to compute current 3D point,  #
            # then store the projected 2D points in the corresponding arrays      #
            # you shoul have something like: x1[n] = ..., y1[n] = ...,            #
            #                                x2[n] = ..., y2[n] = ...             #
            ########################################################################

            # Project 3D points to 2D using perspective projection
            # For camera 1: project X1
            proj1 = K1 @ X1  # Apply camera intrinsic matrix
            x1[n] = proj1[0] / proj1[2]  # Normalize by z-coordinate (perspective division)
            y1[n] = proj1[1] / proj1[2]

            # For camera 2: project X2
            proj2 = K2 @ X2  # Apply camera intrinsic matrix
            x2[n] = proj2[0] / proj2[2]  # Normalize by z-coordinate (perspective division)
            y2[n] = proj2[1] / proj2[2]

            ########################################################################
            #                           END OF YOUR CODE                           #
            ########################################################################

            # Additional check: ensure projected points are within reasonable image bounds
            # This prevents extreme projections that might cause numerical issues
            if (abs(x1[n]) < 10 and abs(y1[n]) < 10 and 
                abs(x2[n]) < 10 and abs(y2[n]) < 10):
                
                # store the 3D point and go to next one                        
                X1_gt[:, n] = X1
                n += 1
    
    # Check if we successfully generated enough points
    if n < n_pts:
        print(f"Warning: Only generated {n} out of {n_pts} requested points after {attempts} attempts")
        # Trim arrays to actual number of generated points
        x1 = x1[:n]
        x2 = x2[:n] 
        y1 = y1[:n]
        y2 = y2[:n]
        X1_gt = X1_gt[:, :n]

    return x1, x2, y1, y2, K1, K2, X1_gt


# this function is to get the points and camera intrinsic matrix for the "batinria" images.
# you don't need to do anything here
def loadPredefinedPts():
    x1 = np.array([
        10.0000,
        92.0000,
        8.0000,
        92.0000,
        289.0000,
        354.0000,
        289.0000,
        353.0000,
        69.0000,
        294.0000,
        44.0000,
        336.0000,
    ])
    x2 = np.array([
        123.0000,
        203.0000,
        123.0000,
        202.0000,
        397.0000,
        472.0000,
        398.0000,
        472.0000,
        182.0000,
        401.0000,
        148.0000,
        447.0000,
    ])
    y1 = np.array([
        232.0000,
        230.0000,
        334.0000,
        333.0000,
        230.0000,
        278.0000,
        340.0000,
        332.0000,
        90.0000,
        149.0000,
        475.0000,
        433.0000,
    ])
    y2 = np.array([
        239.0000,
        237.0000,
        338.0000,
        338.0000,
        236.0000,
        286.0000,
        348.0000,
        341.0000,
        99.0000,
        153.0000,
        471.0000,
        445.0000
    ])
    K1 = np.array([[844.310547, 0, 243.413315], [0, 1202.508301, 281.529236], [0, 0, 1]])
    K2 = np.array([[852.721008, 0, 252.021805], [0, 1215.657349, 288.587189], [0, 0, 1]])
    return x1, x2, y1, y2, K1, K2


# this is an evaluation function, you don't need to do anything here
def testResults(R_gt, T_gt, X1_gt, R, T, X1):
    scale_factor = np.linalg.norm(T_gt) / np.linalg.norm(T)
    T_scaled = T * scale_factor
    X1_scaled = X1 * scale_factor
    homo_gt = np.zeros((4, 4))
    homo_est = np.zeros((4, 4))
    homo_gt[:3, :3] = R_gt
    homo_gt[:3, -1] = T_gt
    homo_gt[-1, -1] = 1
    homo_est[:3, :3] = R
    homo_est[:3, -1] = T_scaled
    homo_est[-1, -1] = 1
    error_pose = np.linalg.norm(logm(homo_gt) - logm(homo_est))
    error_pts = np.linalg.norm(X1_gt - X1_scaled)
    tol = 1e-9
    if error_pose < tol and error_pts < tol:
        print(
            f'Estimation correct, mean error on pose is {error_pose:.2e}, mean error on points is {error_pts:.2e}')
        return True
    elif error_pose > tol and error_pts < tol:
        print(f'pose estimation is wrong with the mean error of {error_pose: .2e}')
    elif error_pose < tol and error_pts > tol:
        print(f'points estimation is wrong with the mean error of {error_pts: .2e}')
    else:
        print(
            f'pose and points estimation is wrong with the mean error of {error_pose:.2e} and {error_pts:.2e} respectively')
    return False


def visualizeReprojection(I1, I2, x1, y1, x2, y2, X1, R, T, K1, K2):
    """
    I1, I2: PIL images
    x1, y1, x2, y2: original image coordinates
    X1: reconstructed 3D points
    R, T: recovered rotation and translation
    K1, K2: camera intrinsic matrices
    """
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))       # Camera 1
    P2 = K2 @ np.hstack((R, T.reshape(3, 1)))                 # Camera 2

    X1_h = np.vstack((X1, np.ones((1, X1.shape[1]))))         # Homogeneous 3D points

    x1_proj = P1 @ X1_h
    x1_proj /= x1_proj[2]
    x2_proj = P2 @ X1_h
    x2_proj /= x2_proj[2]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Image 1
    axs[0].imshow(I1)
    axs[0].plot(x1, y1, 'ro', label='Original')
    axs[0].plot(x1_proj[0], x1_proj[1], 'g+', label='Reprojected')
    axs[0].set_title("Image 1 - Original vs Reprojected")
    axs[0].legend()

    # Image 2
    axs[1].imshow(I2)
    axs[1].plot(x2, y2, 'ro', label='Original')
    axs[1].plot(x2_proj[0], x2_proj[1], 'g+', label='Reprojected')
    axs[1].set_title("Image 2 - Original vs Reprojected")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
