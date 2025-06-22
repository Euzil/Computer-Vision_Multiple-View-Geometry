import numpy as np

from .utils import skewMat


def transformImgCoord(x1, x2, y1, y2, K1, K2):
    # transform the image coordinates
    # assume the image plane is at z = 1
    # what should 3D points be in camera coordinates?
    # input: 2D points in two images (x1, x2, y1, y2), intrinsics K1, K2
    # output: normalized camera coords x1, x2, y1, y2 (each of shape (n_pts,))

    ########################################################################
    # TODO: Implement the transformation with                              #
    # the given camera intrinsic matrices                                  #
    ########################################################################

    # Convert to homogeneous coordinates
    n_pts = len(x1)
    
    # Create homogeneous coordinates for image 1
    pts1_homo = np.vstack([x1, y1, np.ones(n_pts)])
    # Create homogeneous coordinates for image 2
    pts2_homo = np.vstack([x2, y2, np.ones(n_pts)])
    
    # Transform to normalized camera coordinates by applying inverse of intrinsic matrix
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    
    # Transform points
    pts1_norm = K1_inv @ pts1_homo
    pts2_norm = K2_inv @ pts2_homo
    
    # Extract normalized coordinates
    x1 = pts1_norm[0, :]
    y1 = pts1_norm[1, :]
    x2 = pts2_norm[0, :]
    y2 = pts2_norm[1, :]

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return x1, x2, y1, y2


def constructChiMatrix(x1, x2, y1, y2):
    # construct the chi matrix using the kronecker product
    # input: normalized camera coords x1, y1 in image1 and x2, y2 in image2 
    # output: chi matrix of shape (n_pts, 9)
    n_pts = x1.shape[0]
    chi_mat = np.zeros((n_pts, 9))
    for i in range(n_pts):
        ########################################################################
        # TODO: construct the chi matrix by kronecker product                  #
        ########################################################################
        
        # Create homogeneous coordinates for each point
        p1 = np.array([x1[i], y1[i], 1])  # point in image 1
        p2 = np.array([x2[i], y2[i], 1])  # point in image 2
        
        # Use Kronecker product to create the constraint
        # Each row corresponds to: p2^T @ E @ p1 = 0
        # This gives us: (p2 ⊗ p1)^T @ vec(E) = 0
        chi_mat[i, :] = np.kron(p2, p1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
    return chi_mat


def solveForEssentialMatrix(chi_mat):
    # project the essential matrix onto the essential space
    # input: chi matrix - shape (n_pts, 9)
    # output: essential matrix E - shape (3, 3), U, Vt - shape (3, 3),  S - shape (3, 3) diagonal matrix with E = U @ S @ Vt

    ########################################################################
    # TODO: solve the minimization problem to get the solution of E here.  #
    ########################################################################

    # Solve the homogeneous linear system using SVD
    # We want to minimize ||chi_mat @ vec(E)||^2 subject to ||E||_F = 1
    U_chi, S_chi, Vt_chi = np.linalg.svd(chi_mat)
    
    # The solution is the last column of V (or last row of Vt)
    e_vec = Vt_chi[-1, :]  # Eigenvector corresponding to smallest eigenvalue
    
    # Reshape back to 3x3 matrix
    E = e_vec.reshape(3, 3)
    
    # Compute SVD of E
    U, S, Vt = np.linalg.svd(E)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    # ensure the determinant of U and Vt is positive
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    ########################################################################
    # TODO: Project the E to the normalized essential space here,          #
    # don't forget S should be a diagonal matrix.                          #
    ########################################################################    

    # Essential matrix should have two equal singular values and one zero
    # Project to essential manifold: set singular values to [1, 1, 0]
    S_essential = np.array([1.0, 1.0, 0.0])
    
    # Reconstruct the essential matrix
    E = U @ np.diag(S_essential) @ Vt
    
    # Update S to be the projected singular values
    S = S_essential

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return E, U, Vt, np.diag(S)


def constructEssentialMatrix(x1, x2, y1, y2, K1, K2):
    # compute an approximate essential matrix
    # input: 2D points in two images (x1, x2, y1, y2), camera intrinsic matrix K1, K2
    # output: essential matrix E - shape (3, 3),
    #         singular vectors of E: U, Vt - shape (3, 3),
    #         singular values of E: S - shape (3, 3) diagonal matrix, with E = U @ S @ Vt.

    # you need to finish the following three functions
    x1, x2, y1, y2 = transformImgCoord(x1, x2, y1, y2, K1, K2)
    chi_mat = constructChiMatrix(x1, x2, y1, y2)
    E, U, Vt, S = solveForEssentialMatrix(chi_mat)
    return E, U, Vt, S


def recoverPose(U, Vt, S):
    # recover the possible poses from the essential matrix
    # input: singular vectors of E: U, Vt - shape (3, 3),
    #        singular values of E: S - shape (3, 3) diagonal matrix, with E = U @ S @ Vt.
    # output: possible rotation matrices R1, R2 - each of shape (3, 3),
    #         possible translation vectors T1, T2 - each of shape (3,)

    ########################################################################
    # TODO: 1. implement the R_z rotation matrix.                          #
    #          There should be two of them.                                #
    #       2. recover the rotation matrix R                               #
    #          with R_z, U, Vt. (two of them).                             #
    #       3. recover \hat{T} with R_z, U, S                              #
    #          and extract T. (two of them).                               #
    #       4. return R1, R2, T1, T2.                                      #
    ########################################################################

    # Define W matrix (90 degree rotation around z-axis)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    # Two possible rotation matrices
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # Ensure proper rotations (determinant = +1)
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    
    # Translation is the third column of U (up to sign)
    T1 = U[:, 2]
    T2 = -U[:, 2]

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return R1, R2, T1, T2


def reconstruct(x1, x2, y1, y2, R, T):
    # reconstruct the 3D points from the 2D correspondences and (R, T)
    # input:  normalized camera coords in two images (x1, x2, y1, y2), rotation matrix R - shape (3, 3), translation vector T - shape (3,)
    # output: 3D points X1, X2

    n_pts = x1.shape[0]
    X1, X2 = None, None

    ########################################################################
    # TODO: implement the structure reconstruction matrix M.               #
    #  1. construct the matrix M -shape (3 * n_pts, n_pts + 1)             #
    #    which is defined as page18, chapter 5.                            #
    #  2. find the lambda and gamma as explained on the same page.         #
    #     make sure that gamma is positive                                 #
    #  3. generate the 3D points X1, X2 with lambda and (R, T).            #
    #  4. check the number of points with positive depth,                  #
    #     it should be n_pts                                               #
    ########################################################################

    # Use direct linear triangulation method for each point pair
    X1 = np.zeros((3, n_pts))
    
    for i in range(n_pts):
        # Set up the linear system for triangulation
        # We solve: A * X = 0 where X is the 3D point in homogeneous coordinates
        
        # Point in normalized coordinates
        p1 = np.array([x1[i], y1[i], 1])
        p2 = np.array([x2[i], y2[i], 1])
        
        # Camera projection matrices
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])  # [I | 0] for first camera
        P2 = np.hstack([R, T.reshape(3, 1)])           # [R | t] for second camera
        
        # Build the constraint matrix using cross product
        # p1 × (P1 @ X) = 0 and p2 × (P2 @ X) = 0
        A = np.zeros((4, 4))
        
        # First camera constraints (2 equations from cross product)
        A[0] = p1[1] * P1[2] - P1[1]  # y1*P1[2,:] - P1[1,:]
        A[1] = P1[0] - p1[0] * P1[2]  # P1[0,:] - x1*P1[2,:]
        
        # Second camera constraints (2 equations from cross product)  
        A[2] = p2[1] * P2[2] - P2[1]  # y2*P2[2,:] - P2[1,:]
        A[3] = P2[0] - p2[0] * P2[2]  # P2[0,:] - x2*P2[2,:]
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]  # Homogeneous 3D point
        
        # Convert to 3D coordinates
        if abs(X_h[3]) > 1e-10:
            X1[:, i] = X_h[:3] / X_h[3]
        else:
            return None, None  # Point at infinity
    
    # Transform to second camera coordinates
    X2 = R @ X1 + T.reshape(3, 1)
    
    # Check chirality constraint (positive depth)
    n_positive_depth1 = np.sum(X1[2, :] > 0)
    n_positive_depth2 = np.sum(X2[2, :] > 0)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    if n_positive_depth1 == n_pts and n_positive_depth2 == n_pts:
        return X1, X2
    else:
        return None, None


def allReconstruction(x1, x2, y1, y2, R1, R2, T1, T2, K1, K2):
    # reconstruct the 3D points from the 2D correspondences and the possible poses
    # input: 2D points in two images (x1, x2, y1, y2), possible rotation matrices R1, R2 - each of shape (3, 3),
    #        possible translation vectors T1, T2 - each of shape (3,), intrinsics K1, K2
    # output: the correct rotation matrix R, translation vector T, 3D points X1, X2

    num_sol = 0
    #transform to camera coordinates
    x1, x2, y1, y2 = transformImgCoord(x1, x2, y1, y2, K1, K2)
    # first check (R1, T1)
    X1, X2 = reconstruct(x1, x2, y1, y2, R1, T1)
    if X1 is not None:
        num_sol += 1
        R = R1
        T = T1
        X1_res = X1
        X2_res = X2

    # check (R1, T2)
    X1, X2 = reconstruct(x1, x2, y1, y2, R1, T2)
    if X1 is not None:
        num_sol += 1
        R = R1
        T = T2
        X1_res = X1
        X2_res = X2

    # check (R2, T1)
    X1, X2 = reconstruct(x1, x2, y1, y2, R2, T1)
    if X1 is not None:
        num_sol += 1
        R = R2
        T = T1
        X1_res = X1
        X2_res = X2

    # check (R2, T2)
    X1, X2 = reconstruct(x1, x2, y1, y2, R2, T2)
    if X1 is not None:
        num_sol += 1
        R = R2
        T = T2
        X1_res = X1
        X2_res = X2

    if num_sol == 0:
        print('No valid solution found')
        return None, None, None, None
    elif num_sol == 1:
        print('Unique solution found')
        return R, T, X1_res, X2_res
    else:
        print('Multiple solutions found')
        return R, T, X1_res, X2_res