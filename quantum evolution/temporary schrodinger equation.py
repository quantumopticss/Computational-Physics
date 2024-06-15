import numpy as np
import matplotlib.pyplot as plt
import odesolver as ode
from matplotlib.animation import FuncAnimation

def tse_initial(xlist):
    ## function
    k = 3
    # exp((0-1j)*k*x)*exp(-(x-1)**2)
    i_list = np.empty_like(xlist,dtype = complex); 
    N = np.size(xlist)
    i = 0
    while (i<N):
        i_list[i] = np.exp((0+1j)*k*xlist[i])*np.exp(-(xlist[i]-8)**2)
        i += 1 
    return i_list

def tse_odefun(t,x,H):
    ##  dx/dt = i*(H/hbar)x
    dxdt = (0-1j)* (H @ x)
    return dxdt

def tse_funH(xlist,h,hbar,m):
    n = np.size(xlist)
    T = np.zeros([n,n],dtype = complex);V = np.zeros([n,n],dtype = complex)

    ## kkinetic energy
    T[0,[0,1]] = np.array([-2,1])
    T[n-1,[n-2,n-1]] = np.array([1,-2])
    
    i = 1
    while(i<n-1):
        T[i,[i-1,i,i+1]] = np.array([1,-2,1])
        i += 1 

    T = T*(-hbar**2/(2*m*h**2))
    
    ## potential energy
    i = 0
    while(i<n):
        if xlist[i] >= 15:
            V[i,i] = 5
        i += 1
    
    H = T+V
    Vlist = np.empty_like(xlist,dtype = complex)
    i = 0
    while (i<n):
        Vlist[i] = V[i,i]
        i += 1
    return Vlist,H

def tse_main():
    ## users set
    # initial function - to tse_initial
    h = 0.1
    L = 30

    hbar = 1;m = 1
    tspan = [0,4]

    ## operate
    N = int(np.ceil(L/h))
    xlist = np.arange(0,N,1)*h
    psi0 = tse_initial(xlist)  # [0,1,---,N-1]
    Vlist,H = tse_funH(xlist,h,hbar,m)
    H = H/hbar

    ## temporal evolution
    tlist,psi = ode.ode45(tse_odefun,tspan[0],psi0,tspan[1],args = (H,),step_max = 1e-2,TOL = 1e-5)
    rho = np.abs(psi)**2
    ## figure
    fig, ax = plt.subplots()
    time_diff = tlist[1:] - tlist[0:-1:1]

    ani = FuncAnimation(fig, update, frames=len(tlist),fargs = (ax,xlist,tlist,rho,))
    ani.save('probability_density_evolution.gif', writer='imagemagick')
    return 0
    plt.figure(1)
    tn = np.size(tlist)
    i = 0
    while(i<tn-1):
        plt.clf()
        plt.plot(xlist,rho[i,:],label = f"time = {tlist[i]}")
        plt.plot(xlist,Vlist)
        plt.xlim(xlist[0],xlist[-1])
        plt.title('probability density temporary evolution')
        plt.xlabel("x")  # Xlabel
        plt.legend()
        plt.pause(tlist[i+1] - tlist[i])
        i += 1

def update(frame,ax,xlist,tlist,rho):
    ax.clear()
    ax.plot(xlist, rho[frame,:], label=f"time = {tlist[frame]}")
    ax.set_xlim(xlist[0], xlist[-1])
    ax.set_ylim(0,1)
    ax.set_title('Probability Density Temporary Evolution')
    ax.set_xlabel("x")
    ax.legend()

if __name__ == "__main__":
    tse_main()

