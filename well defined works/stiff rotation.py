import numpy as np
import matplotlib.pyplot as plt
import odesolver as ode

def sr_fun(t,w,I):
    beta_1 = (I[1] - I[2])*w[1]*w[2]/I[0]
    beta_2 = (I[2] - I[0])*w[2]*w[0]/I[1]
    beta_3 = (I[0] - I[1])*w[0]*w[1]/I[2]

    dwdt = np.array([beta_1,beta_2,beta_3])
    return dwdt

def sr_main():
    ## users set
    I1 = 1
    I2 = 2
    I3 = 3
    w10 = 0.1
    w20 = 0.2
    w30 = 2
    tspan = [0,10]

    ## operate
    w_initial = np.array([w10,w20,w30])
    I = np.array([I1,I2,I3])

    tlist,wlist = ode.ode23(sr_fun,tspan[0],w_initial,tspan[1],args = (I,))

    ## figure
    plt.figure(1)
    plt.plot(tlist,wlist[:,0],label = "omega-1")
    plt.plot(tlist,wlist[:,1],label = "omega-2")
    plt.plot(tlist,wlist[:,2],label = "omega-3")
    plt.legend()
    plt.xlabel('time')
    plt.title('Omega-t')
    plt.show()

if __name__ == "__main__":
    sr_main()