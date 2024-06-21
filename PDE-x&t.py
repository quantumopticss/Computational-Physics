import numpy as np
import odesolver as ode
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

## Right Boundary: BC I, y = 0
## Left Boundary: BC II, py/pn = 0

def fun(t,Y,c,h):
    dYdt = np.zeros_like(Y)
    N = np.size(Y)
    dYdt[0:N//2-1:1] = Y[N//2:-1]
    dYdt[N//2+1:-1] = c**2*( Y[0:N//2-2] + Y[2:N//2] - 2*Y[1:N//2 - 1] )/h**2
    dYdt[N//2] = dYdt[N//2 + 1]

    return dYdt

def pd_main():
    c = 1
    h = 0.01
    xlist = np.arange(0,202*h+h,h)
    ylist0_y = np.zeros_like(xlist)
    ylist0_y[1:200+1:1] = xlist[1:200+1:1]*np.sin(np.pi*xlist[1:200+1:1])/1.3
    ylist0_vy = np.zeros_like(ylist0_y)

    Y = np.hstack((ylist0_y,ylist0_vy))

    tlist,Y_list = ode.ode23(fun,0,Y,1,args = (c,h,), step_max = 0.1, TOL = 1e-2)

    ylist = Y_list[:,0:202+1:1]
    fig, ax = plt.subplots()
    
    ani = FuncAnimation(fig, update, frames=len(tlist),fargs = (ax,xlist,ylist,tlist))
    ani.save('Oscillating Evolution.gif', writer='imagemagick')


def update(frame,ax,xlist,ylist,tlist):
    ax.clear()
    ax.plot(xlist,ylist[frame,:])
    ax.set_ylim([-1.5,1.5])
    ax.set_title(f'Oscillating Evolution, time = {tlist[frame]}')

if __name__ == "__main__":
    pd_main()