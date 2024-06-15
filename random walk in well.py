import numpy as np
import odesolver as ode
import matplotlib.pyplot as plt
import numpy.random as nr
from matplotlib.animation import FuncAnimation

#def rw_Ffun(x):
#    F = -(x**2 - 3**2)*4*x
#    return F

def rw_fun(t,x,f,T,mu_m,gamma):    
    # input rand number to control the process
    a = (-gamma*x[1] + f(x[0]) )/mu_m + T*2*(nr.rand()-0.5)/mu_m
    dxdt = np.array([x[1],a])

    return dxdt

def rw_main():
    ## users set *********************************
    #V_fun = lambda x: (x**2 - 2)*(x**2 + 2)
    T = 0.4 # K
    mu_m = 20 # kg/mol # m_molecular /NA
    R = 8.31 # kb * NA
    gamma = 8. # ddot{x} = -gamma*dot{x}/m
    V0 = 15*1e-1

    ## operate
    f = lambda x: -4*V0*x*(x**2-2)
    T *= R
    tspan = [0,12]
    x0 = [-1.4,0]

    tlist, xlist = ode.ode00(rw_fun,tspan[0],x0,tspan[1],args = (f,T,mu_m,gamma,),step = 1e-1)
    
    ## figure
    X = np.linspace(-3,3,100)
    V_fun = lambda x: V0*(x**2 - 2)**2
    fig, ax = plt.subplots()

    ani = FuncAnimation(fig, update,frames=len(tlist),interval=10,fargs = (ax,tlist,xlist,X,V_fun)) # interval in ms (0.01s)
    ani.save('Langevin euqation.gif', writer='imagemagick')

def update(frame,ax,tlist,xlist,X,V_fun):
    ax.clear()
    ax.plot(X,V_fun(X))
    vx = V_fun(xlist[frame,0])
    ax.scatter(xlist[frame,0],vx)
    ax.set_title(f'random walk of , time = {tlist[frame]}')

if __name__ == "__main__":
    rw_main()