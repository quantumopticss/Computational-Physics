import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt

def fun(t,x):
    ## x'' = - 5^2 * x

    # x0' = x1
    # x1' = -w^2*x0

    dydt = [x[1],-5**2*x[0]]
    return dydt
    
def main():
    
    y0 = [0,1] # x0 & x'0
    solver = RK45(fun,0,y0,10,max_step = 0.1)
    
    t_values = []
    y0_values = []

    while solver.status == 'running':
        solver.step()
        t_values.append(solver.t)
        y0_values.append(solver.y[0])

    tlist = np.linspace(0,10,1000,endpoint =True)
    y = 0.2*np.sin(5*tlist)
    plt.plot(t_values,y0_values)
    plt.plot(tlist,y)
    plt.show()



if __name__ == "__main__":
    main()