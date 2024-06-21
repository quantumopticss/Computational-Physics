import numpy as np
import matplotlib.pyplot as plt
import odesolver as ode
from matplotlib.animation import FuncAnimation

def ht_fun(t,T,h,k,Q,f1,f2):
    
    (_,N) = np.shape(T);N -= 2
    dTdt = np.zeros_like(T)
    dTdt[1:N+1,1:N+1] = k*(T[1:N+1,2:N+2] + T[1:N+1,0:N] + T[0:N,1:N+1] + T[2:N+2,1:N+1] - 4*T[1:N+1,1:N+1] )/h**2
    dTdt[0,:] = dTdt[1,:]
    dTdt[-1,:] = dTdt[-2,:]
    dTdt[:,0] = dTdt[:,1]
    dTdt[f1:f2,0] = 0
    
    return dTdt

def ht_main():
    ## users set
    L = 1
    Tl = 1
    Tr_p = 0.5
    a = 0.1
    h = 0.01 # mesh
    tspan = [0,0.002] # tspan 
    k = 1 # heat transfer coefficients

    # boundary condition&
    # initial condition
    N = int(np.ceil(L/h))
    f1 = N//2 - int(np.ceil(a//(2*h)))
    f2 = N//2 + int(np.ceil(a//(2*h)))
    T0 = np.zeros([N+2,N+2])
    T0[:,-1] = Tl
    T0[f1:f2,0] = Tr_p

    Q = np.ones([N,N])*(1e-4)

    # operate
    tlist,Tlist = ode.odeii(ht_fun,tspan[0],T0,tspan[1],args = (h,k,Q,f1,f2),order = 2,t_step = 0.0001,TOL = 1e-2)

    ## figure
    fig, ax = plt.subplots()

    ani = FuncAnimation(fig, update, frames=len(tlist),fargs = (ax,tlist,Tlist))
    ani.save('Heat Transfer Evolution.gif', writer='imagemagick')

def update(frame,ax,tlist,T):
    ax.clear()
    ax.imshow(T[frame,...],cmap='viridis')
    ax.set_title(f'Heat Transfer Evolution, time = {tlist[frame]}')

if __name__ == "__main__":
    ht_main()
