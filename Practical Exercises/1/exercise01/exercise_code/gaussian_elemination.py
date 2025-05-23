import numpy as np

def swap_rows(A, i, j):
    '''
    Inputs:
    - A: numpy.ndarray, matrix
    - i: int, index of the first row
    - j: int, index of the second row

    Outputs:
    - numpy.ndarray, matrix with swapped rows
    '''
    A[[i, j]] = A[[j, i]]
    return A

def multiply_row(A, i, scalar):
    '''
    Inputs:
    - A: numpy.ndarray, matrix
    - i: int, index of the row
    - scalar: float, scalar to multiply the row with

    Outputs:
    - numpy.ndarray, matrix with multiplied row
    '''
    A[i] = A[i] * scalar
    return A

def add_row(A, i, j, scalar=1):
    '''
    Inputs:
    - A: numpy.ndarray, matrix
    - i: int, index of the row to be added to
    - j: int, index of the row to be added

    Outputs:
    - numpy.ndarray, matrix with added rows
    '''
    A[i] = A[i] + A[j]*scalar
    return A

def perform_gaussian_elemination(A):
    '''
    Inputs:
    - A: numpy.ndarray, matrix of shape (dim, dim)

    Outputs:
    - ops: List[Tuple[str,int,int]], sequence of elementary operations
    - A_inv: numpy.ndarray, inverse of A
    '''
    dim = A.shape[0]
    A_inv = np.eye(dim)
    ops = []
    ########################################################################
    # TODO:                                                                #
    # Implement the Gaussian elemination algorithm.                        #
    # Return the sequence of elementary operations and the inverse matrix. #
    #                                                                      #
    # The sequence of the operations should be in the following format:    #
    # • to swap to rows                                                    #
    #   ("S",<row index>,<row index>)                                      #
    # • to multiply the row with a number                                  #
    #   ("M",<row index>,<number>)                                         #
    # • to add multiple of one row to another row                          #
    #   ("A",<row index i>,<row index j>, <number>)                        #
    # Be aware that the rows are indexed starting with zero.               #
    # Output sufficient number of significant digits for numbers.          #
    # Output integers for indices.                                         #
    #                                                                      #
    # Append to the sequence of operations                                 #
    # • "DEGENERATE" if you have successfully turned the matrix into a     #
    #   form with a zero row.                                              #
    # • "SOLUTION" if you turned the matrix into the $[I|A −1 ]$ form.     #
    #                                                                      #
    # If you found the inverse, output it as a second element,             #
    # otherwise return None as a second element                            #
    ########################################################################
    A = A.copy().astype(float)
    for col in range(dim):
        # 找到最大主元（pivoting）
        max_row = np.argmax(np.abs(A[col:, col])) + col
        if np.isclose(A[max_row, col], 0):
            ops.append("DEGENERATE")
            return ops, None

        # 如果当前行不是最大行，交换它们
        if max_row != col:
            A = swap_rows(A, col, max_row)
            A_inv = swap_rows(A_inv, col, max_row)
            ops.append(("S", col, max_row))

        # 将主元变为1
        pivot = A[col, col]
        if not np.isclose(pivot, 1.0):
            A = multiply_row(A, col, 1.0 / pivot)
            A_inv = multiply_row(A_inv, col, 1.0 / pivot)
            ops.append(("M", col, 1.0 / pivot))

        # 消去其他行该列
        for row in range(dim):
            if row != col and not np.isclose(A[row, col], 0):
                factor = -A[row, col]
                A = add_row(A, row, col, factor)
                A_inv = add_row(A_inv, row, col, factor)
                ops.append(("A", row, col, factor))

    ops.append("SOLUTION")
    return ops, A_inv

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
