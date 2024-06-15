import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile

# w in THz
# lbd in nm
def dp_Apulse(sigmaG_tau,sigmaL_tau,accuracy = 2048):
    # pulse input
    # U(t) = A(t)*exp(j*2*pi*nu0*t)
    # A(t) is pulse profile
    delta_tau = np.sqrt(sigmaG_tau**2 + sigmaL_tau**2)
    tlist = np.linspace(-20*delta_tau,20*delta_tau,accuracy) # in 1s/1e12
    Alist = 100*voigt_profile(tlist,sigmaG_tau,sigmaL_tau)
    tlist = tlist - tlist[0]

    return tlist, Alist, accuracy
    ## 
    plt.figure(1)
    plt.plot(tlist,np.abs(Ulist))
    plt.show()

def dp_main():
    ## users set ************************************
    a = np.array([2.8939])
    b = np.array([0.13967]) # properties of material SiO2

    z = 0.0000002 # propagation length in m
    accuracy = 4096
    ## incident wave
    nu0 = 200 # in THZ , wavelength = 1500nm
    sigmaG_tau = 1 # 1/1e12
    sigmaL_tau = 0.02
    tlist, Alist, accuracy = dp_Apulse(sigmaG_tau,sigmaL_tau,accuracy) # tlist is sampling time list and Ulist is the correspond U 
    Ulist = Alist*np.exp(1j*2*np.pi*nu0*tlist)

    delta_t = (tlist[-1] - tlist[0])/(accuracy - 1)
    ## fourier transform
    F_nu = np.fft.fftshift(np.fft.fft(Alist))/accuracy
    nu_f = np.fft.fftshift(np.fft.fftfreq(accuracy,delta_t))
    
    ## phase delay & Eamp_attenuation of dispersive media
    phi_delay, Eamp_attenuation = dp_propagtion((nu_f+nu0),z,a,b)

    ## recombination
    A_re = np.zeros_like(tlist,dtype = complex)
    i = 0
    while(i<accuracy):
        A_re = A_re + F_nu[i]*np.exp(1j*2*np.pi*nu_f[i]*tlist)*Eamp_attenuation[i]*np.exp(-1j*phi_delay[i])
        i += 1
    U_re = A_re*np.exp(1j*2*np.pi*nu0*tlist)

    ## figrue
    Fout_nu = np.fft.fftshift(np.fft.fft(A_re))/accuracy
    plt.subplot(2,2,1)
    plt.plot(tlist,np.real(Ulist),label = 'real - amplitude')
    plt.plot(tlist,np.abs(Ulist)**2,label = 'intensity')
    plt.xlabel('time[ps]')
    plt.legend()
    plt.title('pulse incident')

    plt.subplot(2,2,2)
    plt.plot(tlist,np.real(U_re),label = 'real - amplitude')
    plt.plot(tlist,np.abs(U_re)**2,label = 'intensity')
    plt.xlabel('time[ps]')
    plt.legend()
    plt.title('pulse output')

    plt.subplot(2,2,3)
    plt.plot(nu_f+nu0,np.abs(F_nu)**2)
    plt.title('incident frequency component')
    plt.xlabel('frequency [THz]')

    plt.subplot(2,2,4)
    plt.plot(nu_f+nu0,np.abs(Fout_nu)**2)
    plt.title('output frequency component')
    plt.xlabel('frequency [THz]')
    plt.show()

def dp_propagtion(nu_f,z,a,b):

    #alpha = np.ones_like(nu_f)*0.01
    #phi_delay = (12000000*np.pi*(np.linspace(-1,1,np.size(nu_f))**3)) + (51000*np.pi*(np.linspace(-1,1,np.size(nu_f))**2))
    #return phi_delay, Eamp_attenuation

    c0_const = 2.998*1e8
    lbd0 = (c0_const/nu_f)*1e-3
    k0 = 2*np.pi*1e9/lbd0
    
    #kai1, kai2 = dp_material(lbd0,a,b)
    
    nu0 = 200
    delta_nu = 0.1
    kai1 = nu0**2*1.78*(nu0**2 - nu_f**2)/( delta_nu**2*nu_f**2 + (nu0**2 - nu_f**2)**2 )
    kai2 = -nu0**2*1.78*nu_f*delta_nu/( delta_nu**2*nu_f**2 + (nu0**2 - nu_f**2)**2 )

    beta = k0*np.sqrt((1+kai1) + np.sqrt( kai2**2 + (1+kai1)**2  ))/np.sqrt(2)
    alpha = k0*np.sqrt(np.sqrt((kai1+1)**2 + kai2**2) - (1+kai1))/np.sqrt(2)

    phi_delay = beta*z
    Eamp_attenuation = np.exp(-alpha*z)

    return phi_delay, Eamp_attenuation

def dp_material(lbd,a,b):
    ## lbd in nm
    lbd_c2 = (lbd/1e3)**2 # in um and **2 

    #n = 1 + 0.6961663*lbd_c2/(lbd_c2 - 0.0684043**2) + 0.4079426*lbd_c2/(lbd_c2 - 0.1162414**2) + 0.8974794*lbd_c2/(lbd_c2 - 9.896161**2) 
    n2 = np.ones_like(lbd_c2)
    num = np.size(a)
    if num != np.size(b):
        ValueError("size of coefficient a must equal to size of b")
    
    i = 0
    ## sellmeier Eq
    while(i<num):
        n2 = n2 + a[i]*lbd_c2/(lbd_c2 - b[i]**2)
        i += 1

    kai1 = np.sqrt(n2 - 1)
    # return kai1, kai2
    return kai1, np.zeros_like(kai1)

def dp_nplot():
    a = np.array([0.696,0.408,0.897])
    b = np.array([0.068,0.116,9.896])
    lbd = np.arange(200,6300,1)
    n = dp_material(lbd,a,b)
    
    ##
    plt.figure(1)
    plt.plot(lbd,n)
    plt.xlabel('wavelength[nm]')
    plt.title('refractive index of SiO2')
    plt.show() 

if __name__ == "__main__":
    dp_main()