import numpy as np

def lf_main():
    A = np.array([[1,0,1],
                  [0,1,0],
                  [0,1,1]])
    b = np.array([1,2,4])
    x = LU_solver(A,b)
    print(x)

def LU_solver(A,b):
    ## calculate
    x,erc = LU_operator(A,b)

    ## return error or result 
    match erc:
        case 's':
            print('error,A is singular')
            return
    
        case 'a':
            print('error,dimension not match')
            return

        case 'r':
            return x

def LU_operator(A,b):
    ## solve function Ax = b using LU methods
    A = A.astype(np.float64); b = b.astype(np.float64)

    S_A = np.shape(A)
    N = np.size(b)

    if S_A[0] != N or S_A[1] != N:
        return 0,'a'
    
    ## LU algorithm ***************************************
    Matrix = np.zeros([N,N,N])
    i = 0
    while(i<N): # calculate LU
        Matrix[:,:,i],A,erc = lf_operate(A,i,N)
        if erc == 's':
            return 0,'s'
        i += 1

    ## we have the linear function of the formular LUx = b
    # U = A (end)
    # L = muliply (M[:,:,i]) [N-1,N-2,...2,1,0]
    L = Matrix[:,:,0]
    j = 1
    while(j<N):
        L = L @ Matrix[:,:,j]    ## U = M_n @ M_{n-1} @ ... @ M_2 @ M_1 --> L = MI_1 @ MI_2 @ ... @MI_{n-1} @ MI_n
        j += 1

    ## solve LUx = b;  -> Ly = b;Ux = y
    # solve y, Ly = b
    y = np.empty_like(b)
    j = 0
    while(j<N):
        k = 0
        y[j] = b[j]
        while(k<j):
            y[j] = y[j] - y[k]*L[j,k]
            k += 1
        y[j] = y[j]/L[j,j]
        j += 1

    # solve x, Ux = y
    x = np.zeros_like(y)
    j = N-1
    while (j>=0):
        k = N-1
        x[j] = y[j]
        while(k>j):
            x[j] = x[j] - A[j,k]*x[k]
            k -= 1
        x[j] = x[j]/A[j,j]
        j -= 1 

    return x,'r'

def lf_operate(A,k,N):
    M = np.eye(N)
    judge = k
    while(1):
        if A[k,k] != 0:
            M[k,k] = 1/A[k,k]
            break
        else:
            judge += 1
            if judge >= N:
                return 0,0,'s'
            transfer = A[k,:] 
            A[k,:] = A[judge,:]
            A[judge,:] = transfer

    i = 1
    while(i+k<N):
        M[k+i,k] = -A[k+i,k]/A[k,k]
        i += 1
    
    # MI
    #MI = np.linalg.inv(M)
    MI = np.eye(N)
    MI[k,k] = A[k,k]
    i = 1
    while(i+k<N):
        MI[k+i,k] = A[k+i,k]
        i += 1

    A = M @ A
    ## return
    return MI,A,'r'

if __name__ == "__main__":
    lf_main()