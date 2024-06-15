# Kuramoto Model
# d [theta_i]/dt = [omage_i] + K/N \sum_{j} sin(\theta_j - theta_i)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import cmath

def km_f(theta,t,N,K,w):
    dfdt = np.empty(N)
    i = 0
    while(i<N):
        dfdt[i] = w + K*np.sum( np.sin(theta - theta[i]) )/N
        i += 1
    
    return dfdt

def km_operator(N,K,w,theta0):
    ## parameter set
    # N -> dimension
    # K -> coupling parameter
    # w -> omega

    tspan = np.linspace(0,8,400)
    theta = odeint(km_f,theta0,tspan,args=(N,K,w))
    
    return tspan,theta


def km_phase_main():
    ## parameter set
    N = 25
    K = np.linspace(0,0.6,15,endpoint = True)
    omega = 5

    ## calculate
    theta0 = 2*np.pi*np.random.rand(N) # initialize
    num = np.size(K)
    
    rlist = np.empty(num)
    Result_his = np.empty([num,N])

    c = 0
    while(c<num):
        Result = km_operator(N,K[c],omega,theta0)
        Result = Result[-1,:] # the last line, theta result
        Result_his[c,:] = Result
        rlist[c] = np.sqrt(np.sum(np.cos(Result))**2 + np.sum(np.sin(Result))**2)/N
        c += 1

    ## figure
    plt.figure(1)
    plt.plot(K,rlist,label = "phase r - K relation")
    plt.legend()
    plt.title("Kuramoto model")

    plt.figure(2)
    alpha = np.linspace(0,2*np.pi,60,endpoint = True)
    plt.plot(np.cos(alpha),np.sin(alpha),'r')
    plt.scatter(np.cos(Result),np.sin(Result),label = "trace phase")
    plt.legend()
    plt.title(f"distribution_end,K={K[-1]}")

    plt.figure(3)
    plt.plot(np.cos(alpha),np.sin(alpha),'r')
    plt.scatter(np.cos(Result_his[2,:]),np.sin(Result_his[2,:]),label = "trace phase")
    plt.legend()
    plt.title(f"distribution,K={K[2]}")

    plt.show()

def km_change_main():
    ## parameter set
    N = 25
    K = 0.6
    omega = 5

    ## calculate
    theta0 = 2*np.pi*np.random.rand(N) # initialize
    tspan,Result = km_operator(N,K,omega,theta0) # theta process

    plt.figure()
    plt.plot(tspan, (Result[:,0] - Result[:,1]),label = "Difference")
    plt.legend()
    plt.title("Kuramoto time varying")
    plt.show()

if __name__ == "__main__":
    #km_phase_main()
    km_change_main()