import numpy as np

def solve_linear_equation_SVD(D, b):
    '''
    Inputs:
    - D: numpy.ndarray, matrix of shape (m,n)
    - b: numpy.ndarray, vector of shape (m,)
    Outputs:
    - x: numpy.ndarray, solution of the linear equation D*x = b
    - D_inv: numpy.ndarray, pseudo-inverse of D of shape (n,m)
    '''

    ########################################################################
    # TODO:                                                                #
    # Solve the linear equation D*x = b using the pseudo-inverse and SVD.  #
    # Your code should be able to tackle the case where D is singular.     # 
    ########################################################################


    U, S, Vt = np.linalg.svd(D, full_matrices=False)

    # Construct pseudo-inverse of the diagonal matrix S
    S_inv = np.zeros_like(S)
    tolerance = 1e-10  # To avoid division by very small values
    S_inv[S > tolerance] = 1. / S[S > tolerance]
    S_inv_mat = np.diag(S_inv)

    # Compute pseudo-inverse of D
    D_inv = Vt.T @ S_inv_mat @ U.T

    # Compute solution
    x = D_inv @ b

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return x, D_inv

