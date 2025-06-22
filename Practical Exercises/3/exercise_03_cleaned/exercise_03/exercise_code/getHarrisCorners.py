import numpy as np
from scipy.ndimage import maximum_filter

def getHarrisCorners(M, kappa, theta):
    # Compute Harris corners
    # Input:
    # M: structure tensor of shape (H, W, 2, 2)
    # kappa: float (parameter for Harris corner score) 
    # theta: float (threshold for corner detection)
    # Output:
    # score: numpy.ndarray (Harris corner score) of shape (H, W)
    # points: numpy.ndarray (detected corners) of shape (N, 2)

    ########################################################################
    # TODO:                                                                #
    # Compute the Harris corner score and find the corners.               #
    #                                                                      #
    # Hints:                                                               #
    # - The Harris corner score is computed using the determinant and      #
    #   trace of the structure tensor.                                     #
    # - Use the threshold theta to find the corners.                       #
    # - Use non-maximum suppression to find the corners.                   #
    ########################################################################

    # Extract elements of the structure tensor
    M11 = M[:, :, 0, 0]  # Ix^2
    M12 = M[:, :, 0, 1]  # Ix*Iy
    M22 = M[:, :, 1, 1]  # Iy^2
    
    # Compute determinant and trace of M
    det_M = M11 * M22 - M12 * M12  # determinant = M11*M22 - M12^2
    trace_M = M11 + M22            # trace = M11 + M22
    
    # Compute Harris corner response
    # Harris score = det(M) - kappa * trace(M)^2
    score = det_M - kappa * (trace_M ** 2)
    
    # Apply threshold to get potential corners
    corner_candidates = score > theta
    
    # Non-maximum suppression with 3x3 neighborhood
    # Find local maxima
    local_maxima = maximum_filter(score, size=3) == score
    
    # Combine threshold and local maxima conditions
    corners = corner_candidates & local_maxima
    
    # Get corner coordinates
    corner_coords = np.where(corners)
    points = np.column_stack((corner_coords[1], corner_coords[0]))  # (x, y) format
    
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return score, points