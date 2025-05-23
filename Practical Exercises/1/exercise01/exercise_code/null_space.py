import numpy as np
def get_null_vector(D):
    '''
    Inputs:
    - D: numpy.ndarray, matrix of shape (m,n)
    Outputs:
    - null_vector: numpy.ndarray, matrix of shape (dim_kern,n)
    '''

    ########################################################################
    # TODO:                                                                #
    # Get the kernel of the matrix D.                                      #
    # the kernel should consider the numerical errors.                     #
    ########################################################################


    U, S, Vt = np.linalg.svd(D)
    tol = 1e-10  # tolerance for zero singular values
    rank = (S > tol).sum()
    
    # Null space consists of the right singular vectors corresponding to zero singular values
    null_vector = Vt[rank:]  # shape: (dim_kern, n)


    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return null_vector 
