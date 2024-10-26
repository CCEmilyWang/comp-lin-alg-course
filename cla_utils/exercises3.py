import numpy as np


def householder(A):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations. The reduction should be done "in-place",
    so that A is transformed to R.

    :param A: an mxn-dimensional numpy array
    """

    m, n = A.shape
    
    I = np.eye(m)
    # if m >n:
    for i in range(n):
        x= A[i:m,i]
        a = np.sign(x[0])
        a = 1 if x[0] >= 0 else -1
        b=np.linalg.norm(x)
        if b==0 :
            b=1
        Alph = a*b
        e = np.zeros_like(x)
        e[0]=1
        v=x+Alph*e
        v=v/np.linalg.norm(v)
        A[i:m,i:n]=A[i:m,i:n]-2*np.outer(v,(np.dot(v, A[i:m,i:n])))

    # R=A
    # print (A)

    return None  # 指示就地操作


def solve_U(U, b):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """
    m,m = U.shape
    x = np.zeros(m)
    x[m] = b[m]/U[m,m]
    for i in range (m-1,-1,-1):
        x[i] = b[i]
        for j in range (i+1,m):
           x[i]-=x[j]*U[i,j]
        x[i]/= U[i,i]
    return None
                     
    raise NotImplementedError


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """
    m , n = A.shape

    Ahat = np.hstack((A, b)) 
    householder(Ahat)
    x= solve_U(Ahat[0:m,0:n], Ahat[:m,n:])

    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the full QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """
    m, n = A.shape
    R = A.copy()  
    Q = np.eye(m) 
    
    for i in range(n):
       
        x = R[i:, i]
        
        # Create the Householder vector
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x) * (1 if x[0] >= 0 else -1)
        u = x - e
        v = u / np.linalg.norm(u)
        
       
        R[i:, :] -= 2.0 * np.outer(v, np.dot(v, R[i:, :]))
        
      
        Q[:, i:] -= 2.0 * np.outer(np.dot(Q[:, i:], v), v)
    
    return Q, R



def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """
    Q,R = householder_qr(A)


    x= householder_solve(R, Q.conj().T * b)

    return x
