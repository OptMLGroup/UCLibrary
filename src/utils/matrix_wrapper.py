############

#   @File name: matrix_wrapper.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-09-21 16:46:34

#   @Last modified by:  Xi He
#   @Last Modified time:    2018-10-08 13:39:58

#   @Description:
#   @Example:

############

import numpy as np
from numpy import linalg as LA
from scipy import linalg as SLA

def eigDecomp(x, A = None):
    u, v = LA.eig(A)
    return u[-1], np.real(v[-1])

def leftRightMost(A = None):
    # u, _ = LA.eig(A)
    # s_u = sorted(u)
    # return np.real(s_u[0]), np.real(s_u[-1])
    u = SLA.eigh(A, eigvals_only=True, eigvals=(0, A.shape[0]-1))
    return u[0], u[-1]

def rightMost(A = None):
    return SLA.eigh(A, eigvals_only=True, eigvals=(A.shape[0]-1, A.shape[0]-1))

def leftMost(A = None):
    # u, _ = LA.eig(A)
    # return np.real(sorted(u[0]))
    return SLA.eigh(A, eigvals_only=True, eigvals=(0, 0))

def matrixDamping(A, lmd):
    assert A.shape[0] == A.shape[1], "matrix size not match!!"
    return A + lmd * np.eye(A.shape[0])

def isPDMatrix(A):
    try:
        LA.cholesky(A)
        return True
    except LA.LinAlgError as err:
        if 'Matrix is not positive definite' in err.message:
            return False
        else:
            raise err

def isPSDMatrix(A):
    return leftMost(A)[0] >= 0