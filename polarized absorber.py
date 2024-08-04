import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
import numerical_integral as ni
from matplotlib.animation import FuncAnimation

def ab_operate(B,nu0,vec,vec_basic,gamma_up_Lc,gamma_up_L,gamma_up_Rc,nu_up,gamma_down_Lc,gamma_down_L,gamma_down_Rc,nu_down,T_line,T_ab):
    ## calculate the transimittance of each line
    
    # nu_up; nu_down is the frequency of the up and down energy level
    # nu_0 is the frequency of the incident field (a particular line)
    # gamma is the frequency bias of those energy levels
    # vec is basic vectors

    # energy modify and linewidth
    eta = np.empty_like(B)
    N = 7*1e19 # number dnesity of Na ************************************************** retardation of transmittance start
    tsp_Na_line = 9*1e-9 # lifetime of Na 589.3 nm ##****************************************************************** 
    tsp_Na_ab = 11*1e-9

    # parameters
    c_const = 2.998*1e8
    m_Na = 23*1e-3
    d_Na = 0.186*1e-9
    f_col = 4*N*np.pi* (12*d_Na)**2 * np.sqrt(8.31*T_ab/(np.pi*m_Na))

    delta_nu_L_line = 1/(2*np.pi) *(1/tsp_Na_line + 2*f_col)/1e12 # in THz
    delta_nu_L_ab = 1/(2*np.pi) *(1/tsp_Na_ab + 2*f_col)/1e12
    delta_nu_G_ab = nu0*np.sqrt(8.31*T_ab/(m_Na*c_const**2)) # delta for Na in THz
    delta_nu_G_line = nu0*np.sqrt(8.31*T_line/(m_Na*c_const**2)) # delta for sodium lamp in THz
    alpha0 = (N*(c_const/(nu0*1e12))**2/(8*np.pi*tsp_Na_ab) * 1e-2)/1e12 #
    
    # operate
    i = 0
    while(i<np.size(B)):
        nuB_up_L = nu_up + gamma_up_L*B[i]
        nuB_up_Lc = nu_up + gamma_up_Lc*B[i]
        nuB_up_Rc = nu_up + gamma_up_Rc*B[i]
        nuB_down_L = nu_down + gamma_down_L*B[i]
        nuB_down_Lc = nu_down + gamma_down_Lc*B[i]
        nuB_down_Rc = nu_down + gamma_down_Rc*B[i]

        eta[i] = ni.integral45(ab_fun,[508.42,509.05],args = (vec,vec_basic,nu0,nuB_up_L,nuB_up_Lc,nuB_up_Rc,nuB_down_L,nuB_down_Lc,nuB_down_Rc,delta_nu_G_ab,delta_nu_G_line,delta_nu_L_ab,delta_nu_L_line,alpha0),h_max = 0.001,TOL = 1e-6)
        i += 1

    return eta

def ab_abline(nu,nu_B,delta_nu_G_ab,delta_nu_L,delta):
    # generate absorption line of sodium atom
    # nu_B are central frequencies of those lines
    # and each line has voigt_profile
    g = 0
    i = 0
    while (i<np.size(nu_B)):
        g += voigt_profile(nu-nu_B[i],delta_nu_G_ab,delta_nu_L/2)
        i += 1 

    g = g*delta
    return g 

def ab_fun(nu,vec,vec_basic,nu0,nuB_up_L,nuB_up_Lc,nuB_up_Rc,nuB_down_L,nuB_down_Lc,nuB_down_Rc,delta_nu_G_ab,delta_nu_G_line,delta_nu_L_ab,delta_nu_L_line,alpha0): # polarization
    ## based on polarization to calculate lines
    ## Lc = (LPx + iLPy) /sqrt(2) 

    f_i = voigt_profile(nu-nu0,delta_nu_G_line,delta_nu_L_line/2) # lineshape of incident light
    if vec == 'C' and vec_basic == 'C':
        nu_B = np.hstack((nuB_up_Lc,nuB_down_Lc))
        g = ab_abline(nu,nu_B,delta_nu_G_ab,delta_nu_L_ab,1)

    if vec == 'L' and vec_basic == 'L':
        nu_B = np.hstack((nuB_up_L,nuB_down_L))
        g = ab_abline(nu,nu_B,delta_nu_G_ab,delta_nu_L_ab,1)

    if vec == 'C' and vec_basic == 'L':
        nu_B = np.hstack((nuB_up_L,nuB_down_L,nuB_up_L,nuB_down_L))
        g = ab_abline(nu,nu_B,delta_nu_G_ab,delta_nu_L_ab,1/2)

    if vec == 'L' and vec_basic == 'C':
        nu_B = np.hstack((nuB_up_Lc,nuB_down_Lc,nuB_up_Rc,nuB_down_Rc))
        g = ab_abline(nu,nu_B,delta_nu_G_ab,delta_nu_L_ab,1/2)

    f_ab = np.exp(-alpha0*g)
    return f_i*f_ab

def ab_main():
############################################################################ users set
    theta = 0.98*(np.pi/2) # the angle between light and B field 
    # hbar = 6.62*10**(-34)
    # lbd0 = 589.0*10**(-9) & 589.6*1e-9 
    # e = 1.602*10**(-19)
    # me = 9.109*10**(-31)
    gamma_0 = 1.399*1e-2 # THz/T = e/(4*pi*me)
    
    nu_down = 508.480 # THz % 6 lines : Rc RC, L L, Lc Lc
    nu_up = 508.998 # THz % 4 lines : Rc, L L, Lc

    T_ab = 850 # tempurature of Na absorber ###########************************************************************
    T_line = 1700 # tempuarture of sodium lamp
    # N = 10**22
    
    # gammas are used to calculate energy level modified by B 
    gamma_up_Rc = np.array([0.5,5/6])*gamma_0 
    gamma_up_L = np.array([-1/6,1/6])*gamma_0
    gamma_up_Lc = np.array([-0.5,-5/6])*gamma_0

    gamma_down_Rc = np.array([2/3])*gamma_0
    gamma_down_L = np.array([-1/3,1/3])*gamma_0
    gamma_down_Lc = np.array([-2/3])*gamma_0  
    
    ## operate
    B = np.arange(0,1.2+0.1,0.1)
    
    ## assume the radiation of those lines from sodium lamp is equivalent 
    lines_up_L = 2
    lines_down_L = 2
    lines_up_Lc = 4
    lines_down_Lc = 2
    LL = 10
    vec_Lc = 'C'
    vec_L = 'L'

    ## Longitudianl 
    # when the prppagation direction od light is parallel with Magnetfield
    eta_KupL = ab_operate(B,nu_up,vec_L,vec_Lc,gamma_up_Lc,gamma_up_L,gamma_up_Rc,nu_up,gamma_down_Lc,gamma_down_L,gamma_down_Rc,nu_down,T_line,T_ab)
    eta_KupLc = ab_operate(B,nu_up,vec_Lc,vec_Lc,gamma_up_Lc,gamma_up_L,gamma_up_Rc,nu_up,gamma_down_Lc,gamma_down_L,gamma_down_Rc,nu_down,T_line,T_ab)
    eta_KdL = ab_operate(B,nu_down,vec_L,vec_Lc,gamma_up_Lc,gamma_up_L,gamma_up_Rc,nu_up,gamma_down_Lc,gamma_down_L,gamma_down_Rc,nu_down,T_line,T_ab)
    eta_KdLc = ab_operate(B,nu_down,vec_Lc,vec_Lc,gamma_up_Lc,gamma_up_L,gamma_up_Rc,nu_up,gamma_down_Lc,gamma_down_L,gamma_down_Rc,nu_down,T_line,T_ab)

    eta_K = (lines_up_L*eta_KupL + lines_up_Lc*eta_KupLc + lines_down_L*eta_KdL + lines_down_Lc*eta_KdLc)/LL
    ## Transverse
    eta_TupL = ab_operate(B,nu_up,vec_L,vec_L,gamma_up_Lc,gamma_up_L,gamma_up_Rc,nu_up,gamma_down_Lc,gamma_down_L,gamma_down_Rc,nu_down,T_line,T_ab)
    eta_TupLc = ab_operate(B,nu_up,vec_Lc,vec_L,gamma_up_Lc,gamma_up_L,gamma_up_Rc,nu_up,gamma_down_Lc,gamma_down_L,gamma_down_Rc,nu_down,T_line,T_ab)
    eta_TdL = ab_operate(B,nu_down,vec_L,vec_L,gamma_up_Lc,gamma_up_L,gamma_up_Rc,nu_up,gamma_down_Lc,gamma_down_L,gamma_down_Rc,nu_down,T_line,T_ab)
    eta_TdLc = ab_operate(B,nu_down,vec_Lc,vec_L,gamma_up_Lc,gamma_up_L,gamma_up_Rc,nu_up,gamma_down_Lc,gamma_down_L,gamma_down_Rc,nu_down,T_line,T_ab)

    eta_T = (lines_up_L*eta_TupL + lines_up_Lc*eta_TupLc + lines_down_L*eta_TdL + lines_down_Lc*eta_TdLc)/LL

    # total transmittance
    eta = eta_K*(np.cos(theta)**2) + eta_T*(np.sin(theta)**2)

    ## figure
    plt.figure(1)
    plt.plot(B*1000,eta,label = 'transmittance')
    plt.legend()
    plt.xlabel('Magnetic field/mT')
    plt.ylabel('transmittance')
    plt.title(f"theta = {2*theta/np.pi:.2f} pi/2")
    
    plt.show()

def ab_Bline():
    ##
    N = 1e18
    c_const = 2.998*1e8
    tsp_Na_ab = 10*1e-9 
    T_ab = 1000
    nu0 = 508.75

    # parameters
    m_Na = 23*1e-3
    d_Na = 0.186*1e-9
    f_col = 4*N*np.pi*d_Na**2 * np.sqrt(8.31*T_ab/(np.pi*m_Na))

    delta_nu_L_ab = 1/(2*np.pi) *(1/tsp_Na_ab + 2*f_col)/1e12
    delta_nu_G_ab = nu0*np.sqrt(8.31*T_ab/m_Na)/c_const # delta for Na in THz

    ##
    gamma_0 = 1.399*1e-2 # THz/T = e/(4*pi*me)

    gamma_up_L = np.array([-1/6,1/6])*gamma_0
    gamma_up_C = np.array([0.5,5/6,-0.5,-5/6])*gamma_0

    gamma_down_L = np.array([-1/3,1/3])*gamma_0
    gamma_down_C = np.array([2/3,-2/3])*gamma_0 

    B1 = np.linspace(0,0.4,20)
    B2 = np.linspace(0.4,1,10)
    B = np.concatenate((B1,B2),axis = 0)
    Nb = np.size(B)

    nu_down = 508.480 # THz % 6 lines : Rc RC, L L, Lc Lc
    nu_up = 508.998 # THz % 4 lines : Rc, L L, Lc
    nulist = np.linspace(508.97,509.03,10000)
    Nnu = np.size(nulist)

    g = np.zeros([Nb,Nnu])
    i = 0
    while(i<Nb):
        b = B[i]
        nu_up_list = nu_up + gamma_up_L*b 
        nu_down_list = nu_down  + gamma_down_L*b
        
        #nu_Blist = np.concatenate((nu_up_list,nu_down_list),axis = 0)
        nu_Blist = nu_up_list
        for nub in nu_Blist:
            g[i,:] = g[i,:] + voigt_profile(nulist - nub,delta_nu_G_ab,delta_nu_L_ab)
        
        i += 1
            
    ## figure
    fig, ax = plt.subplots()
    anim = FuncAnimation(fig, update, frames = Nb,interval=200,fargs = (ax,nulist,g,B))
    anim.save('MM_Magnetic field for linefunction.gif', writer='imagemagick')

def update(frame,ax,nulist,g,B):
    ax.clear()
    ax.plot(nulist,0.7*g[frame,:],label = 'modulated line',color = 'b')
    ax.plot(nulist,g[0,:],label = 'original line',color = 'r')
    ax.legend()
    ax.set_ylim([0, np.max(g[0,:],axis = None)])
    ax.set_ylabel('lineshape fuinction')
    ax.set_xlabel('frequency [THz]')
    ax.set_title(f'B-field = {B[frame]:.2f}T')  

if __name__ == "__main__":
    ab_main()
    #ab_Bline()
    