import numpy as np
import matplotlib.pyplot as plt
import odesolver as ode

## second ordered nonlinear optics phenomenon: optical parametric conversion
# use optical field of wavelength 312nm and 520nm to generate optical field of wavelength 780nm
# due to special design, phase matching condition is only satisfied at such frequency pair
# so that we do not consider optical fields of other frequencies

def no_main2():
    ## users set ****************
    eta = 377*np.sqrt(2) ## impedence
    hbar_const = 6.62*1e-34/(2*np.pi) ## planck's constant
    c0_const = 2.998*1e8 
    d = 1e-38 ## second ordered nonlinear coefficient 
    a10 = 1e7
    a20 = 7*1e6
    a30 = 0.

    ## operate
    # due to phase matching condition, we only consider w1 + w2 = w3
    # here we choose w1 = (780nm), w2 = (520nm)
    PI = np.pi
    lbd_1 = 780*1e-9
    lbd_2 = 520*1e-9
    w1 = 2*PI*c0_const/lbd_1
    w2 = 2*PI*c0_const/lbd_2
    w3 = w1 + w2
    g = d*np.sqrt(2*eta**3*w1*w2*w3)
    # we set E_field = sqrt(2*eta*hbar_const*w_i)*a_i so that abs(a_i)**2 is photon flux density
    # from maxwell equation we can get the equation that 
    # a3'' = 2jk3[a3' + jg*a1*a2*exp(-j*(deltak)*z)] # deltak = k1 + k2 - k3
    zspan = [0,0.5] # m

    w_list = np.array([w1,w2,w3])
    k_list = w_list*no_nfun(w_list)/c0_const 
    delta_k = k_list[0] + k_list[1] - k_list[2]
    da10 = -1j*g*np.conj(a20)*a30*np.exp(1j*delta_k)
    da20 = -1j*g*a30*np.conj(a10)*np.exp(1j*delta_k)
    da30 = -1j*g*a10*a20*np.exp(-1j*delta_k)

    a_initial = np.array([a10,da10,a20,da20,a30,da30],dtype = complex) # [a1,a1',a2,a2',a3,a3']
    
    zlist , a_list = ode.ode45(no_afun2,zspan[0],a_initial,zspan[1],args = (no_nfun,w1,w2,w3,g),step_max = 1e-2,TOL = 1e-2)

    ## figure**************************************************************
    a1 = a_list[:,0]
    a2 = a_list[:,2]
    a3 = a_list[:,4]

    plt.subplot(1,2,1)
    plt.plot(zlist,np.abs(a1)**2,label = 'phi_1')
    plt.plot(zlist,np.abs(a2)**2,label = 'phi_2')
    plt.plot(zlist,np.abs(a3)**2,label = 'phi_3')
    plt.legend()
    plt.title('photon flux density')

    plt.subplot(1,2,2)
    plt.plot(zlist,hbar_const*w1*np.abs(a1)**2,label = 'I_1')
    plt.plot(zlist,hbar_const*w2*np.abs(a2)**2,label = 'I_2')
    plt.plot(zlist,hbar_const*w3*np.abs(a3)**2,label = 'I_3')
    plt.legend()
    plt.title('Intensity')

    plt.show()

def no_afun2(z,a_list,no_nfun,w1,w2,w3,g):
    
    c0_const = 2.998*1e8
    w_list = np.array([w1,w2,w3])
    k_list = w_list*no_nfun(w_list)/c0_const 

    delta_k = k_list[0] + k_list[1] - k_list[2]
    a1 = a_list[0]
    a2 = a_list[2]
    a3 = a_list[4]

    dda3 = 2*1j*k_list[2]*(a_list[5] + 1j*g*a1*a2*np.exp(-1j*delta_k*z))
    dda1 = 2*1j*k_list[0]*(a_list[1] + 1j*g*np.conj(a2)*a3*np.exp(1j*delta_k*z))
    dda2 = 2*1j*k_list[1]*(a_list[3] + 1j*g*a3*np.conj(a1)*np.exp(1j*delta_k*z))

    dadz = np.array([a_list[1],dda1,a_list[3],dda2,a_list[5],dda3])
    return dadz

def no_main():
    ## users set ****************
    eta = 377*np.sqrt(2) ## impedence
    hbar_const = 6.62*1e-34/(2*np.pi) ## planck's constant
    c0_const = 2.998*1e8 
    d = 1e-36 ## second ordered nonlinear coefficient 
    a10 = 0
    a20 = np.sqrt(8*1e18)
    a30 = np.sqrt(1e20)

    ## operate
    # due to phase matching condition, we only consider w1 + w2 = w3
    # here we choose w1 = (780nm), w2 = (520nm)
    PI = np.pi
    lbd_1 = 780*1e-9
    lbd_2 = 520*1e-9
    w1 = 2*PI*c0_const/lbd_1
    w2 = 2*PI*c0_const/lbd_2
    w3 = w1 + w2
    g = d*np.sqrt(2*eta**3*w1*w2*w3)
    # we set E_field = sqrt(2*eta*hbar_const*w_i)*a_i so that abs(a_i)**2 is photon flux density
    # from maxwell equation we can get the equation that 
    # a3'' = 2jk3[a3' + jg*a1*a2*exp(-j*(deltak)*z)] # deltak = k1 + k2 - k3
    zspan = [0,0.2] # m
    w_list = np.array([w1,w2,w3])

    a_initial = np.array([a10,a20,a30],dtype = complex) # [a1,a1',a2,a2',a3,a3']
    
    zlist , a_list = ode.ode23(no_afun,zspan[0],a_initial,zspan[1],args = (no_nfun,w_list,g),step_max = 1e-3,TOL = 1e-2)

    ## figure**************************************************************
    phi_1 = np.abs(a_list[:,0])**2
    phi_2 = np.abs(a_list[:,1])**2
    phi_3 = np.abs(a_list[:,2])**2

    plt.subplot(1,2,1)
    plt.plot(zlist,phi_1,label = 'phi_1')
    plt.plot(zlist,phi_2,label = 'phi_2')
    plt.plot(zlist,phi_3,label = 'phi_3')
    plt.legend()
    plt.title('photon flux density')

    plt.subplot(1,2,2)
    plt.plot(zlist,hbar_const*w1*phi_1,label = 'I_1')
    plt.plot(zlist,hbar_const*w2*phi_2,label = 'I_2')
    plt.plot(zlist,hbar_const*w3*phi_3,label = 'I_3')
    plt.plot(zlist,hbar_const*(w1*phi_1+w2*phi_2+w3*phi_3),label = 'I_all')
    plt.legend()
    plt.title('Intensity')

    plt.show()

def no_afun(z,a_list,no_nfun,w_list,g):
    
    c0_const = 2.998*1e8
    k_list = w_list*no_nfun(w_list)/c0_const

    delta_k = k_list[0] + k_list[1] - k_list[2]

    a1 = a_list[0]
    a2 = a_list[1]
    a3 = a_list[2]

    da3 = - 1j*g*a1*a2*np.exp(-1j*delta_k*z)
    da1 = - 1j*g*np.conj(a2)*a3*np.exp(1j*delta_k*z)
    da2 = - 1j*g*a3*np.conj(a1)*np.exp(1j*delta_k*z)

    dadz = np.array([da1,da2,da3])
    return dadz

def no_nfun(w):
    # n = n(w0) + dn/dw *(w - w0)
    dndw = 1e-28
    d2ndw2 = 1e-55
    return 2*np.ones_like(w) + dndw*(w-w[0]) + 0.5*d2ndw2*((w-w[0])**2)
    

if __name__ == "__main__":
    no_main()