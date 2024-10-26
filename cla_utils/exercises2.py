import numpy as np


def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """

    raise NotImplementedError

    return r, u


def solve_Q(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """

    raise NotImplementedError

    return x


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    raise NotImplementedError

    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an mxl-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U, for appropriate l.
    """

    raise NotImplementedError

    return Q


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place and returning R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """

    m,n = A.shape
    Q=np.zeros(shape=[m,n],dtype=np.complex128)
    R=np.zeros(shape=[n,n],dtype=np.complex128)
    for i in range(n):
        Q[:, i] = A[:, i].copy()
        for j in range(i):
            R[j, i] = np.vdot(Q[:, j], A[:, i])  # 使用共轭转置
            Q[:, i] -= R[j, i] * Q[:, j]
        
        R[i, i] = np.linalg.norm(Q[:, i])  # 计算范数
        if R[i, i] > 1e-10:  # 避免除以零
        #     
            Q[:, i] /= R[i, i]  # 归一化
    B = np.dot(Q, R)
    print ("Q",Q)
    print ("R",R)
    print ("B",B)
    return Q,R


A = np.array([[1+2j, 2+3j, 3+5j],
              [5+1j, 1+1j, 4+1j],
              [9+2j, 7+1j, 1+2j]])
print(GS_classical(A))
Q,R =np.linalg.qr(A)
print(Q@R,np.linalg.qr(A))




def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and returning
    R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """

    m,n = A.shape
    Q=np.zeros(shape=[m,n],dtype=np.complex128)
    R=np.zeros(shape=[n,n],dtype=np.complex128)
    for i in range(n):
        Q[:, i] = A[:, i].copy()
        for j in range(i):
            R[j, i] = np.vdot(Q[:, j], Q[:, i])  # 使用共轭转置
            Q[:, i] -= R[j, i] * Q[:, j]
        
        R[i, i] = np.linalg.norm(Q[:, i])  # 计算范数
        if R[i, i] > 1e-10:  # 避免除以零
        #     
            Q[:, i] /= R[i, i]  # 归一化
    B = np.dot(Q, R)
    print ("Q",Q)
    print ("R",R)
    print ("B",B)
    return Q,R


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is normalised and orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """

    m,n = A.shape
    Q=np.zeros(shape=[m,n],dtype=np.complex128)
    R=np.eye(shape=[n,n],dtype=np.complex128)
    for i in range(k,n):
        Q[:, i] = A[:, i].copy()
        for j in range(i):
            R[j, i] = np.vdot(Q[:, j], A[:, i])  # 使用共轭转置
            Q[:, i] -= R[j, i] * Q[:, j]
        
        R[i, i] = np.linalg.norm(Q[:, i])  # 计算范数
        if R[i, i] > 1e-10:  # 避免除以零
        #     
            Q[:, i] /= R[i, i]  # 归一化
    B = np.dot(Q, R)
    print ("Q",Q)
    print ("R",R)
    print ("B",B)

    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    A = 1.0*A
    R = np.eye(n, dtype=A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        A[:,:] = np.dot(A, Rk)
        R[:,:] = np.dot(R, Rk)
    R = np.linalg.inv(R)
    return A, R
