import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from scipy.integrate import quad
from matplotlib.animation import FuncAnimation
from units import *
# from matplotlib.animation import FuncAnimation
## calculate the transimittance of each line

# nu_up; nu_down is the frequency of the up and down energy level
# nu_0 is the frequency of the incident field (a particular line)
# gamma is the frequency bias of those energy levels
# vec is basic vectors

# energy modify and linewidth

# parameters
m_Na = 23*1e-3
d_Na = 0.186*1e-9*20

k_0 = 1.399*1e-2 # THz/T = e/(4*pi*me) energy modified by (hbar spin * g)

nu_down = 508.480 # THz 
nu_up = 508.998 # THz 
nulist = np.hstack((np.linspace(508.3,508.61,1000),np.linspace(508.87,509.2,1000)))
lbdlist = c0_const/nulist/1e3

lbd_up = 589.0*nm
lbd_down = 589.6*nm

tsp_up = 16.245*ns
tsp_down = 16.299*ns

def ab_main():
############################################################################ users set 
    N_ab = 8*1e17# number ddensity of sodium atom of absorbing
    T_ab = 800 # tempurature of Na absorber ###########************************************************************
    alpha = np.pi/2 # range in [0,pi], 
    T_lamp = 500 # tempuarture of sodium lamp
    N_lamp = 3*1e19
    d = 0.05*m
    """
    alpha is polarization of incident light,
    E = cos(alpha/2) hat{y} + sin(alpha/2)*exp(1j*beta) hat{z}
    """
    # hbar = 6.62*10**(-34)
    # lbd0 = 589.0*10**(-9) & 589.6*1e-9 
    # e = 1.602*10**(-19)
    # me = 9.109*10**(-31)

    ## emit lines
    # assume that sodium lamp emit natural light
    
    f_col_lamp = np.sqrt(2)*pi*d_Na**2*np.sqrt(8*8.31*T_lamp/(pi*m_Na))*N_lamp
    
    gamma_lamp_down = 1/(4*pi)*(1/tsp_down + 2*f_col_lamp)/1e12
    gamma_lamp_up = 1/(4*pi)*(1/tsp_up + 2*f_col_lamp)/1e12
    
    sigma_down = nu_down*np.sqrt(8.31*T_lamp/m_Na)/c0_const
    sigma_up = nu_down*np.sqrt(8.31*T_lamp/m_Na)/c0_const
    
    g_emit_up = ab_line(nu_up,gamma_lamp_up,sigma_up)
    g_emit_down = ab_line(nu_down,gamma_lamp_down,sigma_down)
    
    G_emit_total = lambda nu: 4/3*(6*g_emit_up(nu)*lbd_up**2/(8*pi*tsp_up) + 4*g_emit_down(nu)*lbd_down**2/(8*pi*tsp_down))
    
    ## absorb lines
    ############################################### operate
    # gammas are used to calculate energy level modified by B 
    k_up_L = np.array([1,5/6])*k_0 
    k_up_R = np.array([-1,-5/6])*k_0
    k_up_pi = np.array([-1/3,1/3])*k_0

    k_down_L = np.array([4/3])*k_0
    k_down_R = np.array([-4/3])*k_0
    k_down_pi = np.array([-2/3,2/3])*k_0
        
    B = np.hstack((np.arange(0,0.3,0.05),np.arange(0.3,1.6,0.1)))
    eta = np.empty_like(B)

    f_col_ab = np.sqrt(2)*pi*d_Na**2*np.sqrt(8*8.31*T_ab/(pi*m_Na))*N_ab
    # g is lineshape dictionary which store information of each lines
    for k in range(len(B)):
        ## line functions
        b = B[k]
        nu_up_pi = nu_up + b*k_up_pi
        nu_down_pi = nu_down + b*k_down_pi
        
        nu_up_L = nu_up + b*k_up_L
        nu_down_L = nu_down + b*k_down_L
        
        nu_up_R = nu_up + b*k_up_R
        nu_down_R = nu_down + b*k_down_R

        gamma_ab_down = 1/(4*pi)*(1/tsp_down + 2*f_col_ab)/1e12
        gamma_ab_up = 1/(4*pi)*(1/tsp_up + 2*f_col_ab)/1e12
    
        sigma_down = nu_down*np.sqrt(8.31*T_lamp/m_Na)/c0_const
        sigma_up = nu_down*np.sqrt(8.31*T_lamp/m_Na)/c0_const
    
        g_ab_up_L = ab_line(nu_up_L,gamma_ab_up,sigma_up)
        g_ab_up_R = ab_line(nu_up_R,gamma_ab_up,sigma_up)
        g_ab_up_pi = ab_line(nu_up_pi,gamma_ab_up,sigma_up)
        
        g_ab_down_L = ab_line(nu_down_L,gamma_ab_down,sigma_down)
        g_ab_down_R = ab_line(nu_down_R,gamma_ab_down,sigma_down)
        g_ab_down_pi = ab_line(nu_down_pi,gamma_ab_down,sigma_down)
        
        G_ab_L_total = lambda nu: 4*(g_ab_up_L(nu)*(c0_const/nu_up)**2/(8*pi*tsp_up) + g_ab_down_L(nu)*(c0_const/nu_down)**2/(8*pi*tsp_down))/1e24
        G_ab_R_total = lambda nu: 4*(g_ab_up_R(nu)*(c0_const/nu_up)**2/(8*pi*tsp_up) + g_ab_down_R(nu)*(c0_const/nu_down)**2/(8*pi*tsp_down))/1e24
        G_ab_pi_total = lambda nu: 4*(g_ab_up_pi(nu)*(c0_const/nu_up)**2/(8*pi*tsp_up) + g_ab_down_pi(nu)*(c0_const/nu_down)**2/(8*pi*tsp_down))/1e24
        
        G_ab_total = lambda nu: G_ab_pi_total(nu)*np.sin(alpha/2)**2 + (G_ab_L_total(nu) + G_ab_R_total(nu))*np.cos(alpha/2)**2/2

        absorption_nu = lambda nu: N_ab*G_ab_total(nu)*d/1e12
        # total transmittance
        
        varphi_0 = lambda nu: nu*G_emit_total(nu)
        varphi_t = lambda nu: nu*G_emit_total(nu)*np.exp(-absorption_nu(nu))
        
        I0 = quad(varphi_0,508.23,508.73)[0] + quad(varphi_0,508.75,509.25)[0]
        I_transmit = quad(varphi_t,508.23,508.73)[0] + quad(varphi_t,508.75,509.25)[0]
        
        eta[k] = I_transmit/I0

    ## figure
    plt.figure(1)
    plt.plot(B,eta,label = 'transmittance')
    plt.legend()
    plt.xlabel('Magnetic field/mT')
    plt.ylabel('transmittance')
    plt.title(f'N = {N_ab}')
    
    plt.figure(2)
    plt.plot(nulist,G_emit_total(nulist),label = 'emit line')
    plt.plot(nulist,G_ab_total(nulist),label = 'absorption line')
    plt.xlabel('wave length')
    plt.ylabel('linshape function')
    plt.legend()
    
    plt.show()
    
def ab_line(nu0,deltaG_nu,deltaL_nu):
    # generate line of given line(c_nu,sigma,gamma) and weight
    # line are central frequencies of those lines weight are their weights
    # nu is parameter
    # and each line has voigt_profile

    if type(nu0) == float:
        g = lambda nu: voigt_profile(nu-nu0,deltaG_nu,deltaL_nu) 
    elif type(nu0) == np.ndarray:
        g = lambda nu: ab_line_func(nu,nu0,deltaG_nu,deltaL_nu)
        
    return g 

def ab_line_func(nu,nu0,deltaG_nu,deltaL_nu):
    g = 0
    
    if type(deltaG_nu) == np.ndarray:
        for i in range(len(nu0)):
            g = g + voigt_profile(nu-nu0[i],deltaG_nu[i],deltaL_nu[i]) 
    else:
        for i in range(len(nu0)):
            g = g + voigt_profile(nu-nu0[i],deltaG_nu,deltaL_nu)
            
    return g 

def ani_abline():
############################################################################ users set 
    N_ab = 8*1e17# number ddensity of sodium atom of absorbing
    T_ab = 5000 # tempurature of Na absorber ###########************************************************************
    alpha = 1*np.pi/3 # range in [0,pi], 
    T_lamp = 5000 # tempuarture of sodium lamp
    N_lamp = 3*1e19

    """
    alpha is polarization of incident light,
    E = cos(alpha/2) hat{y} + sin(alpha/2)*exp(1j*beta) hat{z}
    """
    # hbar = 6.62*10**(-34)
    # lbd0 = 589.0*10**(-9) & 589.6*1e-9 
    # e = 1.602*10**(-19)
    # me = 9.109*10**(-31)

    ## emit lines
    # assume that sodium lamp emit natural light
    
    f_col_lamp = np.sqrt(2)*pi*d_Na**2*np.sqrt(8*8.31*T_lamp/(pi*m_Na))*N_lamp
    
    gamma_lamp_down = 1/(4*pi)*(1/tsp_down + 2*f_col_lamp)/1e12
    gamma_lamp_up = 1/(4*pi)*(1/tsp_up + 2*f_col_lamp)/1e12
    
    sigma_down = nu_down*np.sqrt(8.31*T_lamp/m_Na)/c0_const
    sigma_up = nu_down*np.sqrt(8.31*T_lamp/m_Na)/c0_const
    
    g_emit_up = ab_line(nu_up,gamma_lamp_up,sigma_up)
    g_emit_down = ab_line(nu_down,gamma_lamp_down,sigma_down)
    
    G_emit_total = lambda nu: 4/3*(6*g_emit_up(nu)*lbd_up**2/(8*pi*tsp_up) + 4*g_emit_down(nu)*lbd_down**2/(8*pi*tsp_down))
    
    ## absorb lines
    ############################################### operate
    # gammas are used to calculate energy level modified by B 
    k_up_L = np.array([1,5/6])*k_0 
    k_up_R = np.array([-1,-5/6])*k_0
    k_up_pi = np.array([-1/3,1/3])*k_0

    k_down_L = np.array([4/3])*k_0
    k_down_R = np.array([-4/3])*k_0
    k_down_pi = np.array([-2/3,2/3])*k_0
        
    B = np.hstack((np.arange(0,0.3,0.05),np.arange(0.3,4.3,0.2)))

    f_col_ab = np.sqrt(2)*pi*d_Na**2*np.sqrt(8*8.31*T_ab/(pi*m_Na))*N_ab
    
    # g is lineshape dictionary which store information of each lines
    g_ab_total  = np.empty([len(B),len(nulist)])
    for k in range(len(B)):
        ## line functions
        b = B[k]
        nu_up_pi = nu_up + b*k_up_pi
        nu_down_pi = nu_down + b*k_down_pi
        
        nu_up_L = nu_up + b*k_up_L
        nu_down_L = nu_down + b*k_down_L
        
        nu_up_R = nu_up + b*k_up_R
        nu_down_R = nu_down + b*k_down_R

        gamma_ab_down = 1/(4*pi)*(1/tsp_down + 2*f_col_ab)/1e12
        gamma_ab_up = 1/(4*pi)*(1/tsp_up + 2*f_col_ab)/1e12
    
        sigma_down = nu_down*np.sqrt(8.31*T_lamp/m_Na)/c0_const
        sigma_up = nu_down*np.sqrt(8.31*T_lamp/m_Na)/c0_const
    
        g_ab_up_L = ab_line(nu_up_L,gamma_ab_up,sigma_up)
        g_ab_up_R = ab_line(nu_up_R,gamma_ab_up,sigma_up)
        g_ab_up_pi = ab_line(nu_up_pi,gamma_ab_up,sigma_up)
        
        g_ab_down_L = ab_line(nu_down_L,gamma_ab_down,sigma_down)
        g_ab_down_R = ab_line(nu_down_R,gamma_ab_down,sigma_down)
        g_ab_down_pi = ab_line(nu_down_pi,gamma_ab_down,sigma_down)
        
        G_ab_L_total = lambda nu: 4*(g_ab_up_L(nu)*(c0_const/nu_up)**2/(8*pi*tsp_up) + g_ab_down_L(nu)*(c0_const/nu_down)**2/(8*pi*tsp_down))/1e24
        G_ab_R_total = lambda nu: 4*(g_ab_up_R(nu)*(c0_const/nu_up)**2/(8*pi*tsp_up) + g_ab_down_R(nu)*(c0_const/nu_down)**2/(8*pi*tsp_down))/1e24
        G_ab_pi_total = lambda nu: 4*(g_ab_up_pi(nu)*(c0_const/nu_up)**2/(8*pi*tsp_up) + g_ab_down_pi(nu)*(c0_const/nu_down)**2/(8*pi*tsp_down))/1e24
        
        G_ab_total = lambda nu: G_ab_pi_total(nu)*np.sin(alpha/2)**2 + (G_ab_L_total(nu) + G_ab_R_total(nu))*np.cos(alpha/2)**2/2
        g_ab_total[k,:] = G_ab_total(nulist)
        
    ## figure
    fig, ax = plt.subplots()
    anim = FuncAnimation(fig, update, frames = len(B),interval=200,fargs = (ax,G_emit_total,g_ab_total,B))    
    anim.save('MM_Magnetic field for linefunction.gif', writer='imagemagick')

def update(frame,ax,G1,g2,B):
    ax.clear()
    ax.plot(nulist,G1(nulist),label = 'original line',color = 'b')
    ax.plot(nulist,g2[frame,:],label = 'modulated line',color = 'r')
    ax.legend()
    ax.set_ylim([0, np.max(G1(nulist),axis = None)])
    ax.set_ylabel('lineshape fuinction')
    ax.set_xlabel('frequency [THz]')
    ax.set_title(f'B-field = {B[frame]:.2f}T')  
############################################

if __name__ == "__main__":
    ani_abline() # spectrum linefunction animiation
    #ab_main() # main
