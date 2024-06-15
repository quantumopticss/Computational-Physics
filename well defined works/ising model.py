import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

def is_main_Cv():
    ## users set  *************************
    Tlist = np.linspace(0.2,6.2,10,endpoint = True)
    J = 1 ## interaction strength
    H = 0 ## outside field
    L = 10 # size of the mesh
    N_rex = 40000
    N_sample = 40000
    sample_freq = 2

    _E = np.array([[],[]])
    _M = np.array([])
    for T in Tlist:
        A = is_operate(J,T,H,L,N_rex,N_sample,sample_freq) ## return a list
        E_ave, E2_ave, M_ave = is_statistic(A,J)
        _M = np.concatenate((_M,np.array([M_ave])),axis = 0)
        _E = np.concatenate((_E,np.array([[E_ave],[E2_ave]])),axis = 1)

    _Cv = (_E[1,:] - (_E[0,:]**2))/T
    ## figure
    plt.subplot(1,2,1)
    plt.plot(Tlist,_M)
    plt.xlabel('Tempurature')
    plt.title('Magnetization')

    plt.subplot(1,2,2)
    plt.plot(Tlist,_Cv)
    plt.xlabel('Tempurature')
    plt.title('Thermal Capacity')

    plt.show()

def is_main_Kai():
    ## users set  *************************
    T = 1.5 ## tempuarture
    J = 1 ## interaction strength
    Hlist = np.linspace(0,2,10) ## outside field
    L = 10 # size of the mesh
    N_rex = 40000
    N_sample = 40000
    sample_freq = 2

    _E = np.array([[],[]])
    _M = np.array([])
    for h in Hlist:
        A = is_operate(J,T,h,L,N_rex,N_sample,sample_freq) ## return a list
        E_ave, E2_ave, M_ave, M2_ave = is_statistic(A,J)
        _M = np.concatenate((_M,np.array([M_ave])),axis = 0)
        _E = np.concatenate((_E,np.array([[E_ave],[E2_ave]])),axis = 1)

    _Cv = (_E[1,:] - (_E[0,:]**2))/Hlist
    ## figure
    plt.subplot(1,2,1)
    plt.plot(Hlist,_M)
    plt.xlabel('Tempurature')
    plt.title('Magnetization')

    plt.subplot(1,2,2)
    plt.plot(Tlist,_Cv)
    plt.xlabel('Tempurature')
    plt.title('Thermal Capacity')

    plt.show()

def is_statistic(A,J):
    ## return E_ave,E2_ave for A
    M = np.sum(A,axis = None)
    dimension = np.shape(A)
    rx = dimension[1]
    ry = dimension[2]
    M = np.abs(M)/(dimension[0]*rx*ry)

    Elist = np.empty([2,dimension[0]])
    i = 0
    while(i<dimension[0]):
        A_i = A[i,...]
        
        auxi = np.empty([rx+2,ry+2])
        auxi[1:-1,1:-1] = A_i
        auxi[0,:] = auxi[rx,:]
        auxi[-1,:] = auxi[1,:]
        auxi[:,0] = auxi[:,ry]
        auxi[:,-1] = auxi[:,1]

        j = 1
        E = 0
        while(j<=rx):
            k = 1
            while(k<=ry):
                E += auxi[j,k]*(auxi[j-1,k] + auxi[j+1,k] + auxi[j,k-1] + auxi[j,k+1])
                k += 1
            j += 1  

        E = E*(-J)/2
        Elist[:,i] = np.array([E,E**2])
        i += 1

    Elist =  np.sum(Elist,axis = 1)/dimension[0]
    return Elist[0], Elist[1], M

def is_operate(J,T,H,L,N_rex,N_sample,sample_freq):
    # initial
    A = 2*rd.randint(2,size = (L+2,L+2)) -1
    A[0,:] = A[L,:]
    A[-1,:] = A[1,:]
    A[:,0] = A[:,L]
    A[:,-1] = A[1,:]

    ## operate
    i = 0
    # relax
    while(i<N_rex):
        nx = rd.randint(1,L+1)
        ny = rd.randint(1,L+1)
        ## E_i = -J*sigma_i*sigma_rs - H*sigma_i
        delta_E = 2*J*A[nx,ny]*(A[nx-1,ny] + A[nx+1,ny] + A[nx,ny-1] + A[nx,ny+1]) + 2*A[nx,ny]*H 
        if delta_E < 0:
            A[nx,ny] = A[nx,ny]*(-1)
        elif rd.rand() < np.exp(-delta_E/T):
            A[nx,ny] = A[nx,ny]*(-1)
        
        i += 1

    ResultA = np.array([A[1:-1,1:-1]]) 
    i = 1
    while(i<N_sample):
        nx = rd.randint(1,L+1)
        ny = rd.randint(1,L+1)

        delta_E = 2*J*A[nx,ny]*(A[nx-1,ny] + A[nx+1,ny] + A[nx,ny-1] + A[nx,ny+1])
        if delta_E < 0:
            A[nx,ny] = A[nx,ny]*(-1)
        elif rd.rand() < np.exp(-delta_E/T):
            A[nx,ny] = A[nx,ny]*(-1)
        
        i += 1
        if i%sample_freq == 0:
            ResultA = np.concatenate((ResultA,np.array([A[1:-1,1:-1]])),axis = 0)
    
    return ResultA

if __name__ == "__main__":
    is_main_Cv()



