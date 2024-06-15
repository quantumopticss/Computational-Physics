import numpy as np
import matplotlib.pyplot as plt

def fe_main():
    ## users set
    A = 10
    B = 10
    h = 0.1 # mesh
    erf = 1e-5# relative erf

    # boundary condition
    Nx = int(np.ceil(A/h))
    Ny = int(np.ceil(B/h))
    T_up = 1*np.ones(Nx+2)
    T_down = 1*np.ones(Nx+2)
    T_left = -1*np.ones(Ny+2)
    T_right = -1*np.ones(Ny+2)
    T = np.zeros([Nx+2,Ny+2])

    rho = np.zeros([Nx,Ny])
    #rho[int(Nx/4):int(Nx/3)+1:1,int(Ny/4):int(Ny/3)+1:1] = 0.1
    #rho[int(2*Nx/3):int(3*Nx/4)+1:1,int(2*Ny/3):int(3*Ny/4)+1:1] = -0.1

    ## calculate
    # initial condition
    T[0,:] = T_up
    T[:,0] = T_left
    T[-1,:] = T_down
    T[:,-1] = T_right
    auxi = np.ones_like(rho)

    i = 1
    while(i<=Nx):
        j = 1
        while(j<=Ny):
            T[i,j] = ((Nx+1-i)*T_up[j] + i*T_down[j])/(Nx+1) + ((Ny+1-j)*T_left[i] + j*T_right[i])/(Ny+1)
            j += 1
        i += 1

    # 
    k = 0
    while(k<1000):
        T_n = 0.25*(T[1:-1,2:Ny+1+1] + T[2:Nx+1+1,1:-1] + T[0:Nx,1:-1] + T[1:-1,0:Ny]   ) + rho*h**2/4
        T[1:-1,1:-1] = T_n
        k += 1

    eps = 1
    while(eps > erf):
        T_n = 0.25*(T[1:-1,2:Ny+1+1] + T[2:Nx+1+1,1:-1] + T[0:Nx,1:-1] + T[1:-1,0:Ny]   ) + rho*h**2/4
        T[1:-1,1:-1] = T_n
        eps = np.sum( np.abs(T[1:-1,1:-1] - T_n)**2/np.max((auxi,np.abs(T_n))) ,axis = None)

    ## figure
    plt.figure(1)
    plt.imshow(T)
    plt.show()

if __name__ == "__main__":
    fe_main()