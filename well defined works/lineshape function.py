## lineshape function simulation
# here we consider three different mechanics which contains to kinds of broadening methods
# 
# lifetime broadening, collision broadening which are homogeneous broadening and are correlated with lorentzian lineshape
# doppler broadening, which is a kinds of inhomogeneous broadening, is correlated with gaussian lineshape 

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.special import erfinv
from numpy.fft import fft

def lf_operator(nu0,tau,f_col,T,m,N):
    ## simulation

    # we must use normalized frequency 1
    
    # doppler broadening:
    # nu' = nu*(1+v/c)
    # v --> f(v) = sqrt(m/2*pi*kb*T)*exp(-0.5*mv^2/kT)
    PI = np.pi
    c_const = 2.998*(10**(8))

    result = np.empty([N,3]) # [[nu,T],[],[],...]
    i = 0

    while(i<N):
        # simulate of lifetime & collision
        T = 0

        # lifetime
        p_life = rand()
        t = tau*np.log(1/(1-p_life)) # expected lifetime
        phi = 2*PI*nu0*t

        # collision
        n_col = t*f_col
        pp_life = np.exp((-1)*n_col) # the possibility of undergo no collision
        p_col = rand()

        if p_col >= pp_life:
            delta_phi = PI*2*(rand() - 0.5)
            phi += delta_phi
            
        # sumilate of doppler broadening
        p_v = rand() # generate velocity of the atom
        while (p_v == 0):
            p_v = rand()
        v = np.sqrt((2*T)/m)*erfinv(p_v)
        
        p_s = rand()
        if p_s > 0.5:
            v *= -1

        nu = nu0*(1+v/c_const) # calculate frequency and phase as a result of doppler broadening
        
        # store data
        result[i,:] = [nu,phi,t]
        i += 1

    ## calculate & return
    accuracy = int(50*N)
    time = np.arange(0,1,1/accuracy) # without endpoint
    E_field = np.zeros_like(time) 

    i = 0
    while(i<N):
        E_field = E_field + np.cos(2*PI*result[i,0]*time+result[i,1])*np.exp(-(time-result[i,2])/(tau*1e12))
        i += 1
    
    E_field = E_field/N

    F_E = fft(E_field)
    freqs = np.fft.fftfreq(len(F_E), 1/accuracy)
    #freqs = freqs*nu0

    positive_freqs = freqs[:len(freqs)//2]
    positive_FE = F_E[:len(F_E)//2]

    positive_freqs = freqs[1:2*nu0:1]
    positive_FE = F_E[1:2*nu0:1]

    return positive_freqs, positive_FE

def lf_main():
    ### users set
    # parameters:
    f_col = 1e7 ## Hz # the frequency at which each atom under goes elastic collision
    tau = 10 ## ns # lifetime of the energy level
    T = 7000 # K  # tempurature of the environment
    m = 20 ## g/mol # the mass of the molecular
    nu0 = 200 # THz
    N = 1000 # rounds to simulate

    # physics constants: 
    R_const = 8.31 ## boltzmann's constant

    ### operate
    # physical parameters
    m = m*(10**(-3)) # true mass of a molecular
    T = T*R_const
    tau = tau*(1e-9) # turn lifetime to dimension of s
    
    # calculate
    freqs,F_E = lf_operator(nu0,tau,f_col,T,m,N)
    F_E = np.abs(F_E)
    n = np.size(F_E)
    average = 0
    i = average
    G_E = np.zeros_like(F_E)
    while(i<=n-1-average):
        G_E[i] = np.sum(F_E[i-average:i+average+1:1],0)/5
        i += 1
        
    ### figure
    plt.figure(1)
    plt.plot(freqs,abs(G_E))
    plt.title('lineshpae function')
    plt.show()

if __name__ == "__main__":
    lf_main()