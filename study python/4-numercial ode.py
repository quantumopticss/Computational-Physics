import numpy as np
import matplotlib.pyplot as plt

# numercial attenuation
def na_main():
    ## users set
    N0 = 100; alpha = 0.2

    tau = 1
    tspan = [0,10]
    
    # function:
    # dN/dt = - alpha*N

    ## operate
    t = np.arange(tspan[0],tspan[-1]+tau,tau)
    N1 = np.empty_like(t)
    N2 = np.empty_like(t)
    N3 = np.empty_like(t)
    N1[0] = N0;  N2[0] = N0; N3[0] = N0

    n = np.size(t)
    # N1 -- Euler Method
    i = 1
    while (i < n):
        N1[i] = N1[i-1]*(1-alpha*tau)
        i += 1

    # N2 -- two step method
    i = 1
    while (i < n):
        N2[i] = N2[i-1] - 0.5*alpha*(N2[i-1] + N2[i-1]*(1-alpha*tau) )*tau
        i += 1


    ## figure
    plt.figure(1)
    plt.plot(t,N1,label = 'A1')
    plt.plot(t,N2,label = 'A2')
    #plt.plot(t,N3,label = 'A3')
    plt.plot(t,N0*np.exp(-alpha*t),label = 'analytical')
    plt.legend()
    plt.title('attenuation, ode with different accuracy')
    plt.show()

if __name__ == "__main__":
    na_main()