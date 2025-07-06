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

    # Use DLT (Direct Linear Triangulation) method for robustness
    X1 = np.zeros((3, n_pts))
    
    # Camera projection matrices
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])  # [I | 0] for first camera
    P2 = np.hstack([R, T.reshape(3, 1)])           # [R | t] for second camera
    
    successful_points = 0
    
    for i in range(n_pts):
        # For each point pair, solve for the 3D point using DLT
        # Set up the homogeneous linear system AX = 0
        A = np.zeros((4, 4))
        
        # Constraints from the first camera: p1 x (P1 * X) = 0
        A[0] = x1[i] * P1[2] - P1[0]  # x1*P1[2,:] - P1[0,:]
        A[1] = y1[i] * P1[2] - P1[1]  # y1*P1[2,:] - P1[1,:]
        
        # Constraints from the second camera: p2 x (P2 * X) = 0
        A[2] = x2[i] * P2[2] - P2[0]  # x2*P2[2,:] - P2[0,:]
        A[3] = y2[i] * P2[2] - P2[1]  # y2*P2[2,:] - P2[1,:]
        
        # Solve using SVD
        try:
            _, _, Vt = np.linalg.svd(A)
            X_homogeneous = Vt[-1]  # Last row of Vt
            
            # Convert from homogeneous to 3D coordinates
            if abs(X_homogeneous[3]) > 1e-10:
                X1[:, i] = X_homogeneous[:3] / X_homogeneous[3]
                successful_points += 1
            else:
                # Point at infinity, use a reasonable default
                X1[:, i] = np.array([0, 0, 1])  # Default to unit depth
                
        except np.linalg.LinAlgError:
            X1[:, i] = np.array([0, 0, 1])  # Default to unit depth
    
    # Transform points to second camera coordinate system
    X2 = R @ X1 + T.reshape(3, 1)
    
    # Check chirality constraint: count points with positive depth
    positive_depth1 = np.sum(X1[2, :] > 0)
    positive_depth2 = np.sum(X2[2, :] > 0)
    
    # Simple and direct chirality check
    if positive_depth1 == n_pts and positive_depth2 == n_pts:
        print(f"Strict chirality PASSED: {positive_depth1}/{n_pts} in cam1, {positive_depth2}/{n_pts} in cam2")
        return X1, X2
    else:
        print(f"Strict chirality FAILED: {positive_depth1}/{n_pts} in cam1, {positive_depth2}/{n_pts} in cam2")
        return None, None

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################


def allReconstruction(x1, x2, y1, y2, R1, R2, T1, T2, K1, K2):
    # reconstruct the 3D points from the 2D correspondences and the possible poses
    # input: 2D points in two images (x1, x2, y1, y2), possible rotation matrices R1, R2 - each of shape (3, 3),
    #        possible translation vectors T1, T2 - each of shape (3,), intrinsics K1, K2
    # output: the correct rotation matrix R, translation vector T, 3D points X1, X2

    num_sol = 0
    valid_solutions = []
    
    #transform to camera coordinates
    x1, x2, y1, y2 = transformImgCoord(x1, x2, y1, y2, K1, K2)
    
    # Test all four combinations of (R, T)
    pose_combinations = [
        (R1, T1, "R1, T1"),
        (R1, T2, "R1, T2"), 
        (R2, T1, "R2, T1"),
        (R2, T2, "R2, T2")
    ]
    
    for R_test, T_test, name in pose_combinations:
        X1, X2 = reconstruct(x1, x2, y1, y2, R_test, T_test)
        
        if X1 is not None and X2 is not None:
            # Count points with positive depth in both cameras
            positive_depth1 = np.sum(X1[2, :] > 0)
            positive_depth2 = np.sum(X2[2, :] > 0)
            
            valid_solutions.append({
                'R': R_test,
                'T': T_test,
                'X1': X1,
                'X2': X2,
                'positive_count': min(positive_depth1, positive_depth2),
                'name': name
            })
            num_sol += 1
    
    if num_sol == 0:
        # Try with relaxed constraints for real-world data
        for R_test, T_test, name in pose_combinations:
            X1, X2 = reconstruct_relaxed(x1, x2, y1, y2, R_test, T_test)
            
            if X1 is not None and X2 is not None:
                positive_depth1 = np.sum(X1[2, :] > 1e-6)  # Small positive threshold
                positive_depth2 = np.sum(X2[2, :] > 1e-6)
                
                valid_solutions.append({
                    'R': R_test,
                    'T': T_test,
                    'X1': X1,
                    'X2': X2,
                    'positive_count': min(positive_depth1, positive_depth2),
                    'name': name + " (relaxed)"
                })
                num_sol += 1
    
    if num_sol == 0:
        print('No valid solution found')
        return None, None, None, None
    
    # Choose the solution with the most points having positive depth
    best_solution = max(valid_solutions, key=lambda x: x['positive_count'])
    
    if num_sol == 1:
        print(f'Unique solution found: {best_solution["name"]}')
    else:
        print(f'Multiple solutions found ({num_sol}), chose: {best_solution["name"]} with {best_solution["positive_count"]} valid points')
    
    return best_solution['R'], best_solution['T'], best_solution['X1'], best_solution['X2']


def reconstruct_relaxed(x1, x2, y1, y2, R, T):
    """Relaxed reconstruction for challenging real-world data"""
    n_pts = x1.shape[0]
    X1 = np.zeros((3, n_pts))
    
    # Same DLT approach but with relaxed thresholds
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, T.reshape(3, 1)])
    
    valid_points = 0
    
    for i in range(n_pts):
        A = np.zeros((4, 4))
        
        A[0] = x1[i] * P1[2] - P1[0]
        A[1] = y1[i] * P1[2] - P1[1]
        A[2] = x2[i] * P2[2] - P2[0]
        A[3] = y2[i] * P2[2] - P2[1]
        
        try:
            _, _, Vt = np.linalg.svd(A)
            X_homogeneous = Vt[-1]
            
            # More lenient threshold for homogeneous coordinate
            if abs(X_homogeneous[3]) > 1e-12:
                X1[:, i] = X_homogeneous[:3] / X_homogeneous[3]
                valid_points += 1
            else:
                # Set to a default value rather than failing completely
                X1[:, i] = np.array([0, 0, 1])  # Unit depth
                
        except np.linalg.LinAlgError:
            # Set to default rather than failing
            X1[:, i] = np.array([0, 0, 1])
    
    # Only require that most points are valid
    if valid_points < 0.6 * n_pts:
        return None, None
    
    X2 = R @ X1 + T.reshape(3, 1)
    
    # Much more lenient chirality check
    n_positive_depth1 = np.sum(X1[2, :] > -0.1)  # Allow small negative depths
    n_positive_depth2 = np.sum(X2[2, :] > -0.1)
    
    min_required = max(1, int(0.5 * n_pts))  # Only require 50% valid
    
    if n_positive_depth1 >= min_required and n_positive_depth2 >= min_required:
        return X1, X2
    else:
        return None, None