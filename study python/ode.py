import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def dydx(x,t,omega,w):
    omega = omega+w
    # \dot{q} = p 
    # \dot{p} = -\omega^2 * q
    p = x[0]
    q = x[1]
    dydx = [(-1)*omega**2*q,p]
    return dydx

def ode_main():

    omega = 1    
    x0 = np.array([1,0])
    t = np.linspace(0,7,1000,endpoint = True)
    x = odeint(dydx,x0,t,args = (omega,omega,))

    plt.figure()
    plt.plot(t,x[:,0],label = "P")
    plt.plot(x[:,0],x[:,1],label = "phase figure")
    plt.legend()
    plt.title("harmonic odeint solver")
    plt.show()

if __name__ == "__main__":
    ode_main()