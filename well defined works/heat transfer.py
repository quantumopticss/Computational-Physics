import numpy as np
import matplotlib.pyplot as plt
import odesolver as ode
from matplotlib.animation import FuncAnimation

def ht_fun(t,T,h,k,Q):
    
    dTdt = np.zeros_like(T)
    A = k*(T[1:-1,0:-2] + T[1:-1,2:] + T[0:-2,1:-1] + T[2:,1:-1] - 4*T[1:-1,1:-1])/h**2 + Q
    dTdt[1:-1,1:-1] = A
    
    return dTdt

def ht_main():
    ## users set
    A = 3
    B = 3
    h = 0.2 # mesh
    tspan = [0,1] # tspan 
    k = 1 # heat transfer

    # boundary condition
    Nx = int(np.ceil(A/h))
    Ny = int(np.ceil(B/h))
    T_up = 1*np.ones(Nx+2)
    T_down = 1*np.ones(Nx+2)
    T_left = -1*np.ones(Ny+2)
    T_right = -1*np.ones(Ny+2)
    T = np.empty([Nx+2,Ny+2])

    Q = np.zeros([Nx,Ny])
    #rho[int(Nx/4):int(Nx/3)+1:1,int(Ny/4):int(Ny/3)+1:1] = 0.1
    #rho[int(2*Nx/3):int(3*Nx/4)+1:1,int(2*Ny/3):int(3*Ny/4)+1:1] = -0.1

    ## calculate
    # initial condition
    T[0,:] = T_up
    T[:,0] = T_left
    T[-1,:] = T_down
    T[:,-1] = T_right
    T_in = np.zeros([Nx,Ny])
    T[1:-1,1:-1] = T_in

    # operate
    tlist,Tlist = ode.ode23(ht_fun,tspan[0],T,tspan[1],args = (h,k,Q,))

    ## figure
    fig, ax = plt.subplots()
    time_diff = tlist[1:] - tlist[0:-1:1]

    ani = FuncAnimation(fig, update, frames=len(tlist),fargs = (ax,tlist,Tlist))
    ani.save('Heat Transfer Evolution.gif', writer='imagemagick')

def update(frame,ax,tlist,T):
    ax.clear()
    ax.imshow(T[frame,:,:])
    ax.set_title(f'Heat Transfer Evolution, time = {tlist[frame]}')

if __name__ == "__main__":
    ht_main()
