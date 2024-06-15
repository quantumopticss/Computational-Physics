import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
import numerical_integral as ni

def ab_lorentz(nu,nu0,delta_nu_L):
    g = delta_nu_L/(2*np.pi*( (nu-nu0)**2 + (delta_nu_L/2)**2   ))
    return g

def ab_fun(nu,nu0,nu_B,delta_nu_G_ab,delta_nu_G_line,delta_nu_L,alpha0): # polarization

    f_i = voigt_profile(nu-nu0,delta_nu_G_line,delta_nu_L*np.sqrt((1/np.log(2))-1)/2) # lineshape of incident light

    i = 0
    g = 0 # lineshape of abs
    while (i<np.size(nu_B)):
        g += voigt_profile(nu-nu_B[i],delta_nu_G_ab,delta_nu_L/2)
        i += 1 

    g = g/1e12
    f_ab = np.exp(-alpha0*g)
    return f_i*f_ab

def ab_main():
    ############################################################################ users set
    theta = 0.99*(np.pi/2) # the angle between light and B field 
    # hbar = 6.62*10**(-34)
    # lbd0 = 589.0*10**(-9) & 589.6*1e-9 
    # e = 1.602*10**(-19)
    # me = 9.109*10**(-31)
    gamma_0 = 1.399*1e-2 # THz/T = e/(4*pi*me)
    
    nu_down = 508.480 # THz % 6 lines : Rc RC, L L, Lc Lc
    nu_up = 508.998 # THz % 4 lines : Rc, L L, Lc

    T_ab = 800 # tempurature of Na absorber ###########**************************************************************************************************** maximun hight 2
    T_line = 600 # tempuarture of sodium lamp
    # N = 10**22
    
    # gammas are used to calculate energy level modified by B 
    gamma_up_Rc = np.array([0.5,5/6])*gamma_0 
    gamma_up_L = np.array([-1/6,1/6])*gamma_0
    gamma_up_Lc = np.array([-0.5,-5/6])*gamma_0

    gamma_down_Rc = np.array([2/3])*gamma_0
    gamma_down_L = np.array([-1/3,1/3])*gamma_0
    gamma_down_Lc = np.array([-2/3])*gamma_0  
    
    ## operate
    B = np.arange(0,2+0.1,0.2)
    
    ## assume the radiation of those lines from sodium lamp is equivalent 
    lines_up_L = 2
    lines_down_L = 2
    lines_up_Lc = 4
    lines_down_Lc = 2

    ## Longitudianl 
    # when the prppagation direction od light is parallel with Magnetfield
    eta_up_Lc_K = ab_operate(gamma_up_Lc,gamma_down_Lc,nu_up,nu_up,nu_down,B,T_ab,T_line)
    eta_down_Lc_K = ab_operate(gamma_up_Lc,gamma_down_Lc,nu_down,nu_up,nu_down,B,T_ab,T_line)

    eta_K = ((lines_down_Lc+lines_down_L)*eta_down_Lc_K + (lines_up_Lc+lines_up_L)*eta_up_Lc_K)/(lines_up_Lc+lines_down_Lc+lines_up_Lc+lines_down_Lc)

    ## Transverse 
    # when the prppagation direction od light is orthogonal with Magnetfield
    eta_up_L_T = ab_operate(gamma_up_L,gamma_down_L,nu_up,nu_up,nu_down,B,T_ab,T_line)
    eta_down_L_T = ab_operate(gamma_up_L,gamma_down_L,nu_down,nu_up,nu_down,B,T_ab,T_line)

    eta_T = ((lines_up_Lc+lines_up_L)*eta_up_L_T + (lines_down_Lc+lines_down_L)*eta_down_L_T )/(lines_up_Lc+lines_down_Lc+lines_up_L+lines_down_L)

    eta = eta_T*np.sin(theta)**2 + eta_K*np.cos(theta)**2

    ## figure
    plt.figure(1)
    plt.plot(B*1000,eta)
    plt.xlabel('Magnetic field/mT')
    plt.ylabel('transmittance')
    plt.title(f"theta = {theta}")
    plt.show()

def ab_operate(gamma_up,gamma_down,nu0,nu_up,nu_down,B,T_ab,T_line): # calculate the transmittance of a particular line
    # nu_up; nu_down is the frequency of the up and down energy level
    # nu_0 is the frequency of the incident field (a particular line)
    # gamma is the frequency bias of those energy levels

    # energy modify and linewidth
    eta = np.empty_like(B)
    
    # parameters
    c_const = 2.998*1e8
    m_Na = 23*1e-3
    N = 6*1e17 # number dnesity of Na ************************************************** retardation of transmittance start
    d_Na = 0.186*1e-9
    tsp_Na = 10*1e-9 # lifetime of Na 589.3 nm ##****************************************************************** maximun hight 1
    f_col = 4*N*np.pi*d_Na**2 * np.sqrt(8.31*T_ab/(np.pi*m_Na))

    delta_nu_L = 1/(2*np.pi) *(1/tsp_Na + 2*f_col)/1e12 # in THz
    delta_nu_G_ab = nu0*np.sqrt(8.31*T_ab/(m_Na*c_const**2)) # delta for Na in THz
    delta_nu_G_line = nu0*np.sqrt(8.31*T_line/(m_Na*c_const**2)) # delta for sodium lamp in THz
    alpha0 = N*(c_const/(nu0*1e12))**2/(8*np.pi*tsp_Na) * 1e-2 #
    
    # operate
    i = 0
    while(i<np.size(B)):
        nu0_B_up = nu_up + gamma_up*B[i]
        nu_B_down = nu_down + gamma_down*B[i]
        nu_B = np.hstack((nu0_B_up,nu_B_down))

        eta[i] = ni.integral45(ab_fun,[508.42,509.05],args = (nu0,nu_B,delta_nu_G_ab,delta_nu_G_line,delta_nu_L,alpha0),h_step = 0.01,TOL = 1e-6)
        i += 1

    return eta

def ab_lines():
    ## users set
    sigma_G = 5
    sigma_L = 1
    
    x = np.arange(-20,20+0.2,0.2)
    y = voigt_profile(x,sigma_G,sigma_L)
    G_x = np.exp(-(x)**2/(2*sigma_G**2))/( np.sqrt(2*np.pi)*sigma_G**2 )
    L_x = sigma_L/(np.pi * (x**2+sigma_L**2))

    ## figure
    plt.figure(1)
    plt.plot(x,y,label = f'voigt,sigma_G = {sigma_G};sigma_L = {sigma_L}')
    plt.plot(x,G_x,label = f'gaussian,sigma_G = {sigma_G};sigma_L = {0}')
    plt.plot(x,L_x,label = f'Lorentzian,sigma_G = {0};sigma_L = {sigma_L}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ab_lines()
