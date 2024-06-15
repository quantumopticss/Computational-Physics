import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt

def nv_fun(t,y):
    # F(t) = sin()
    # beta = 0.01
    F = 20
    beta = 0.5
    w = 3
    dydt = [y[1],F*np.cos(t) - beta*y[1] - w**2*np.sin(y[0])]
    return dydt

def nv_main():
    ## phase figure
    thetalist = np.arange(0.1,0.2,0.1)

    ## figure
    plt.figure(1)
    for theta in thetalist:
        tlist,ylist = nv_operator(theta,'plot')
        if tlist == (-1):
            return 
        plt.plot(tlist,ylist)
    plt.title('phase figrue')
    plt.show() 
    


def nv_operator(theta_start,target = 'return'):
    ## users set

    # function:
    # y'' = - w^2 * sin(y)
    
    tspan = [0,500]
    y_initial = [theta_start,0] # [y0,y0']

    ## solver
    solver = RK45(nv_fun,tspan[0],y_initial,tspan[-1],max_step = 0.1)

    tlist = np.array([0])
    ylist = np.array([0])
    dylist = np.array([0])    

    while solver.status == 'running':
        solver.step() 
        tlist = np.hstack((tlist,solver.t))
        ylist = np.hstack((ylist,solver.y[0]))
        dylist = np.hstack((dylist,solver.y[1]))

    #mod_y = np.fix(ylist/(2*np.pi))
    #ylist = ylist - mod_y*2*np.pi

    ## return
    if target == 'return':
        return  ylist,dylist

    if target == 'plot':
        # figrue
        plt.figure(1)
        plt.plot(tlist,ylist)
        plt.title('time domain y-t')
        plt.show()
        
        plt.figure(2)
        plt.plot(ylist,dylist)
        plt.title('phase domain y - \dot{y}')
        plt.show()

        return -1,-1

if __name__ == "__main__":
    nv_main()
    