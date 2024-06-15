import numpy as np
import odesolver as ode
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

## 2D
def tb_fun(t,R,m1,m2,m3):
    # v
    v = R[6:]

    # a
    r1 = R[0:1+1:1]
    r2 = R[2:3+1:1]
    r3 = R[4:5+1:1]

    a1 = m2*(r2-r1)/( (np.sum( (r2-r1)**2 ))**1.5) + m3*(r3-r1)/( (np.sum( (r3-r1)**2 ))**1.5)
    a2 = m3*(r3-r2)/( (np.sum( (r3-r2)**2 ))**1.5) + m1*(r1-r2)/( (np.sum( (r1-r2)**2 ))**1.5)
    a3 = m1*(r1-r3)/( (np.sum( (r1-r3)**2 ))**1.5) + m2*(r2-r3)/( (np.sum( (r2-r3)**2 ))**1.5)

    ## stack
    a = np.hstack((a1,a2,a3))
    dRdt = np.hstack((v,a))

    return dRdt

def tb_main():
    ## users set
    # paramaters
    # G = 1
    m1 = 0.75
    m2 = 1
    m3 = 1
    tspan = [0,4]

    # initial condition
    r1 = np.array([0,0],dtype = float)
    v1 = np.array([0,0],dtype = float)
    r2 = np.array([1,0],dtype = float)
    v2 = np.array([0,1],dtype = float)
    r3 = np.array([-1,0],dtype = float)
    v3 = np.array([0.5,-1.7],dtype = float)

    ## operate
    R0 = np.hstack((r1,r2,r3,v1,v2,v3))
    tlist,Rlist = ode.ode45(tb_fun,tspan[0],R0,tspan[1],args = (m1,m2,m3,),step_max = 0.01,TOL = 1e-5)

    ## figure
    xlist = np.vstack((Rlist[:,0],Rlist[:,2],Rlist[:,4]))
    ylist = np.vstack((Rlist[:,1],Rlist[:,3],Rlist[:,5]))

    i = np.size(tlist)
    fig, ax = plt.subplots()

    ani = FuncAnimation(fig, update, frames= i,fargs = (ax,xlist,ylist,tlist,)) # interval = 50 
    ani.save('2D three body.gif', writer='imagemagick')
    

def update(frame,ax,xlist,ylist,timelist):
    ax.clear()
    x1 = xlist[0,0:frame+1:1]
    y1 = ylist[0,0:frame+1:1]
    x2 = xlist[1,0:frame+1:1]
    y2 = ylist[1,0:frame+1:1]
    x3 = xlist[2,0:frame+1:1]
    y3 = ylist[2,0:frame+1:1]
    
    ax.plot(x1, y1, label = "M1",color = 'red')
    ax.plot(x2, y2, label = "M2",color = 'blue')
    ax.plot(x3, y3, label = "M3",color = 'green')
    ax.scatter(x1[-1],y1[-1],color = 'red',s = 0.8)
    ax.scatter(x2[-1],y2[-1],color = 'blue',s = 0.8)
    ax.scatter(x3[-1],y3[-1],color = 'green',s = 0.8)

    a = np.min([np.min(xlist,axis = None),np.min(ylist,axis = None)])
    A = np.max([np.max(xlist,axis = None),np.max(ylist,axis = None)])
    ax.set_xlim(a-1, A+1)
    ax.set_ylim(a-1, A+1)

    ax.set_title(f"2D three body, time = {timelist[frame]}")
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.legend()

if __name__ == "__main__":
    tb_main()