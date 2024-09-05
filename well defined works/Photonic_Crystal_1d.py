import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

## computation of 1D photonic crystal
## dispersion relation and E-field distribution

def pc_main():
    ## users set
    n1 = 1.5
    n2 = 3.5
    d1 = 1
    d2 = 1
    N_samp = 2**(7)
    
    ## operation
    num = 3
    klist = np.concatenate((np.arange(-0.485,-0.005+0.06,0.06),np.arange(0.005,0.485+0.06,0.06)),axis = 0)
    ylist = np.empty([num,len(klist)])
    
    nfun = lambda x: n1*(x <= d1)*(x>=0) + n2*(x > d1)*(x<=(d2+d1))
    
    xlist = np.linspace(0,d1+d2,2*N_samp,endpoint = False)
    xlist_plot = np.linspace(0,d1+d2,N_samp,endpoint = False)
    eta_list = 1/(nfun(xlist)**2)

    f_eta = np.fft.fftshift(np.fft.fft(eta_list))/(2*N_samp)
    
    F_eta = f_eta[N_samp]*np.eye(N_samp,k=0,dtype = np.complex128)
    
    for i in range(1,N_samp):
        F_eta = F_eta + f_eta[N_samp + i]*np.eye(N_samp,k=i,dtype = np.complex128) + f_eta[N_samp - i]*np.eye(N_samp,k=-i,dtype = np.complex128)
    
    ## different F matrix for different x
    for order in range(len(klist)):
        f_n = np.arange(N_samp) - N_samp//2 + klist[order]
        F_n = np.tensordot(f_n,f_n,axes = 0)
        
        F = F_n*F_eta
        eig_val, eig_vec = eigh(F)
        
        if klist[order] == 0.005:
            eig_list_a = eig_vec[N_samp//2+2,:]
            eig_list_b = eig_vec[N_samp//2+1,:]
        ylist[:,order] = np.sqrt(eig_val[0:num])
        
    ## plot
    P_a = np.fft.ifft(np.fft.fftshift(eig_list_a))*N_samp
    P_b = np.fft.ifft(np.fft.fftshift(eig_list_b))*N_samp
    plt.figure(1)
    plt.plot(xlist,nfun(xlist),label = "refractive index")
    plt.plot(xlist,(np.fft.ifft(np.fft.fftshift(f_eta))*N_samp*2)**(-0.5),label = "fft refractive index")
    plt.plot(xlist_plot,np.abs(P_a),label = "E-field magnitude_a")
    plt.plot(xlist_plot,np.abs(P_b),label = "E-field magnitude_b")
    plt.xlabel('x')
    plt.legend()
    plt.title("refractive index & field distribution")
    
    plt.figure(2)
    for i in range(num):
        plt.plot(klist,ylist[i,:],label = f"numerical dispersion relation - {i+1}")
    plt.xlabel('Bloch Wavevector in g')
    plt.title('numerical omega in omega_b') # omega_b := 2*pi/(d1+d2)
    plt.legend()
    
    plt.figure(3)
    a = 2*np.pi/(d1+d2)
    wlist_1 = np.linspace(0.001,0.156,25)
    wlist_2 = np.linspace(0.24,0.349,25)
    wlist_3 = np.linspace(0.452,0.595,25)
    K_bloch = lambda w: 1/(2*np.pi)*np.arccos( (n1+n2)**2/(4*n1*n2) * ( np.cos(w*a*(n1*d1+n2*d2)) - (n1-n2)**2/(n1+n2)**2 * np.cos(w*a*(n1*d1 - n2*d2)) )      )    
    plt.plot(K_bloch(wlist_1),wlist_1,label = "ana dispersion relation - 1")
    plt.plot(K_bloch(wlist_2),wlist_2,label = "ana dispersion relation - 2")
    plt.plot(K_bloch(wlist_3),wlist_3,label = "ana dispersion relation - 3")
    plt.xlabel('Bloch Wavevector in g')
    plt.title('analytical omega in omega_b')
    plt.legend()
    
    plt.show()
if __name__ == "__main__":
    pc_main()