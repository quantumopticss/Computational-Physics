import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from scipy.integrate import quad
from matplotlib.animation import FuncAnimation
from collections import defaultdict

def ab_fun(nu,subS,G,delta_nu_L_emit,delta_nu_L_ab,delta_nu_G_emit,delta_nu_G_ab,alpha0,alg = "ave_g"):

    # parameters
    Sline = subS["line"]
    Sweight = subS["weight"]
    Gline_pi = G["pi_line"]
    Gweight_pi = G["pi_weight"]
    Gline_sigma = G["sigma_line"]
    Gweight_sigma =G["sigma_weight"] 

    ## lines of emittion and absorption 
    g_emit = ab_line(nu,Sline,Sweight,delta_nu_G_emit,delta_nu_L_emit)
    g_ab_pi = ab_line(nu,Gline_pi,Gweight_pi,delta_nu_G_ab,delta_nu_L_ab)
    g_ab_sigma = ab_line(nu,Gline_sigma,Gweight_sigma,delta_nu_G_ab,delta_nu_L_ab)
    
    # average 
    if alg == "ave_g":
        g_ab = 0.5*(g_ab_pi + g_ab_sigma)
        g_ab = g_ab*alpha0

        eta = g_emit*np.exp(-g_ab)
        return eta
    
    if alg == "ave_t":
        fun = lambda theta: np.exp(-alpha0*(g_ab_pi*(np.cos(theta))**2 + g_ab_sigma*(np.sin(theta))**2))
        t_ave = (quad(fun,0,np.pi/2))[0]/(np.pi/2)
        return (g_emit*t_ave)

def ab_operate(nu0,subS,G,alg = "ave_g"):
    ## calculate the transimittance of each line
    
    # nu_up; nu_down is the frequency of the up and down energy level
    # nu_0 is the frequency of the incident field (a particular line)
    # gamma is the frequency bias of those energy levels
    # vec is basic vectors

    # energy modify and linewidth
    T_ab = 1000 # tempurature of Na absorber ###########************************************************************
    T_lamp = 600 # tempuarture of sodium lamp
    #N = 5*1e17 # number dnesity of Na ************************************************** retardation of transmittance start
    N = 6*1e17

    N_lamp = 3*1e19
    tsp_Na_line = 10*1e-9 # lifetime of Na 589.3 nm ##****************************************************************** 
    tsp_Na_ab = 10*1e-9

    # parameters
    c_const = 2.998*1e8
    m_Na = 23*1e-3
    d_Na = 0.186*1e-9
    f_col_ab = 4*N*np.pi*d_Na**2 * np.sqrt(8.31*T_ab/(np.pi*m_Na))
    f_col_lamp = 4*N_lamp*np.pi*d_Na**2 * np.sqrt(8.31*T_ab/(np.pi*m_Na))

    delta_nu_L_emit = 1/(2*np.pi) *(1/tsp_Na_line + 2*f_col_lamp)/1e12 # in THz
    delta_nu_L_ab = 1/(2*np.pi) *(1/tsp_Na_ab + 2*f_col_ab)/1e12
    delta_nu_G_ab = nu0*np.sqrt(8.31*T_ab/(m_Na*c_const**2)) # delta for Na in THz
    delta_nu_G_emit = nu0*np.sqrt(8.31*T_lamp/(m_Na*c_const**2)) # delta for sodium lamp in THz
    
    # alpha = alpha0 * g
    alpha0 = (N*(c_const/(nu0*1e12))**2/(8*np.pi*tsp_Na_ab) * 1e-2)/1e12 #
    subeta = quad(ab_fun,508.0,509.5,args = (subS,G,delta_nu_L_emit,delta_nu_L_ab,delta_nu_G_emit,delta_nu_G_ab,alpha0,alg))[0]
    return subeta

def ab_line(nulist,line,weight,deltaG_nu,deltaL_nu):
    # generate line of given line(c_nu,sigma,gamma) and weight
    # line are central frequencies of those lines weight are their weights
    # nu is parameter
    # and each line has voigt_profile
    i = 0
    g = 0
    while (i<np.size(line)):
        g = g + weight[i]*voigt_profile(nulist-line[i],deltaG_nu,deltaL_nu)
        i += 1 

    return g 

def ab_main():
############################################################################ users set 
    # hbar = 6.62*10**(-34)
    # lbd0 = 589.0*10**(-9) & 589.6*1e-9 
    # e = 1.602*10**(-19)
    # me = 9.109*10**(-31)
    alg = "ave_t"
    gamma_0 = 1.399*1e-2 # THz/T = e/(4*pi*me)
    
    nu_down = 508.480 # THz % 6 lines : Rc RC, L L, Lc Lc
    nu_up = 508.998 # THz % 4 lines : Rc, L L, Lc
    # N = 10**22
    
    # gammas are used to calculate energy level modified by B 
    gamma_up_Lc = np.array([0.5,5/6])*gamma_0 
    gamma_up_Rc = np.array([-0.5,-5/6])*gamma_0
    gamma_up_Lz = np.array([-1/6,1/6])*gamma_0
    gamma_up_Lx = np.concatenate((gamma_up_Rc,gamma_up_Lc),axis = 0) # 1/2 

    gamma_down_Lc = np.array([2/3])*gamma_0
    gamma_down_Rc = np.array([-2/3])*gamma_0
    gamma_down_Lz = np.array([-1/3,1/3])*gamma_0
    gamma_down_Lx = np.concatenate((gamma_down_Rc,gamma_down_Lc),axis = 0) # 1/2

    ## emit lines
    # assume that sodium lamp emit natural light
    S_up = defaultdict(list)
    S_up['line'] = nu_up*np.ones([6])
    S_up['weight'] = np.ones([6])

    S_down = defaultdict(list)
    S_down['line'] = nu_down*np.ones([4])
    S_down['weight'] = np.ones([4])

    ############################################### operate
    B = np.arange(0,1.2+0.1,0.1) 
    eta = np.empty_like(B)
    ## Transverse
    # G is lineshape dictionary which store information of each lines
    # G[key] represent a particulare line [freqs,weight]
    k = 0
    while(k<np.size(B)):
        b = B[k]
        nu_up_pi = nu_up + b*gamma_up_Lz
        nu_up_sigma = nu_up + b*gamma_up_Lx
        weight_up_pi = np.ones_like(nu_up_pi)
        weight_up_sigma = np.ones_like(nu_up_sigma)

        nu_down_pi = nu_down + b*gamma_down_Lz
        nu_down_sigma = nu_down + b*gamma_down_Lx
        weight_down_pi = np.ones_like(nu_down_pi)
        weight_down_sigma = np.ones_like(nu_down_sigma)

        G = defaultdict(list)
        G['sigma_line'] = np.concatenate((nu_up_sigma,nu_down_sigma),axis = 0)
        G['sigma_weight'] = np.concatenate((weight_up_sigma,weight_down_sigma),axis = 0)
        G['pi_line'] = np.concatenate((nu_up_pi,nu_down_sigma),axis = 0)
        G['pi_weight'] = np.concatenate((weight_up_pi,weight_down_pi),axis = 0)

        # total transmittance
        eta_up = ab_operate(nu_up,S_up,G,alg)
        eta_down = ab_operate(nu_down,S_down,G,alg)

        eta[k] = (eta_up + eta_down)/(np.sum(S_down["weight"]) + np.sum(S_up["weight"]))
        k += 1

    ## figure
    plt.figure(1)
    plt.plot(B,eta,label = 'transmittance')
    plt.legend()
    plt.xlabel('Magnetic field/mT')
    plt.ylabel('transmittance')
    
    plt.show()

############################################
def ab_Bline():
    ##
    N = 1e18
    c_const = 2.998*1e8
    tsp_Na_ab = 10*1e-9 
    T_ab = 1000
    nu0 = 508.75
    gamma_0 = 1.399*1e-2 # THz/T = e/(4*pi*me)

    gamma_up_Lc = np.array([0.5,5/6])*gamma_0 
    gamma_up_Rc = np.array([-0.5,-5/6])*gamma_0

    gamma_up_pi = np.array([-1/6,1/6])*gamma_0
    gamma_up_sigma = np.concatenate((gamma_up_Rc,gamma_up_Lc),axis = 0) # 1/2 

    weight_up_sigma = 0.5*np.ones_like(gamma_up_sigma)
    weight_up_pi =  np.ones_like(gamma_up_pi)

    # parameters
    m_Na = 23*1e-3
    d_Na = 0.186*1e-9
    f_col = 4*N*np.pi*d_Na**2 * np.sqrt(8.31*T_ab/(np.pi*m_Na))

    delta_nu_L_ab = 1/(2*np.pi) *(1/tsp_Na_ab + 2*f_col)/1e12
    delta_nu_G_ab = nu0*np.sqrt(8.31*T_ab/m_Na)/c_const # delta for Na in THz

    ##
    B1 = np.linspace(0,0.4,20)
    B2 = np.linspace(0.4,1,10)
    B = np.concatenate((B1,B2),axis = 0)
    Nb = np.size(B)

    nu_up = 508.998 #
    nulist = np.linspace(508.97,509.03,10000)
    Nnu = np.size(nulist)

    g_pi = np.zeros([Nb,Nnu])
    g_sigma = np.zeros([Nb,Nnu])
    i = 0
    while(i<Nb):
        b = B[i]
        nuB_up_pi = nu_up + gamma_up_pi*b
        nuB_up_sigma = nu_up + gamma_up_sigma*b

        g_pi[i,:] = ab_line(nulist,nuB_up_pi,weight_up_pi,delta_nu_G_ab,delta_nu_L_ab)
        g_sigma[i,:] = ab_line(nulist,nuB_up_sigma,weight_up_sigma,delta_nu_G_ab,delta_nu_L_ab)
        
        i += 1
     
    ## figure
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    anim = FuncAnimation(fig, update, frames = Nb,interval=200,fargs = (ax1,ax2,ax3,nulist,g_pi,g_sigma,B))
    anim.save('B field for linefunction.gif', writer='imagemagick')

def update(frame,ax1,ax2,ax3,nulist,g_pi,g_sigma,B):
    ax1.clear()
    ax1.plot(nulist,g_pi[frame,:],label = 'pi-modulated line',color = 'b')
    ax1.plot(nulist,g_pi[0,:],label = 'original line',color = 'r')
    ax1.legend()
    ax1.set_ylim([0, 1.2*np.max(g_pi[0,:],axis = None)])
    ax1.set_ylabel('pi-lineshape fuinction')
    ax1.set_xlabel('frequency [THz]')
    ax1.set_title(f'B-field = {B[frame]:.2f}T')  

    ax2.clear()
    ax2.plot(nulist,g_sigma[frame,:],label = 'sigma-modulated line',color = 'b')
    ax2.plot(nulist,g_sigma[0,:],label = 'original line',color = 'r')
    ax2.legend()
    ax2.set_ylim([0, 1.2*np.max(g_sigma[0,:],axis = None)])
    ax2.set_ylabel('sigma-lineshape fuinction')
    ax2.set_xlabel('frequency [THz]')
    ax2.set_title(f'B-field = {B[frame]:.2f}T')  

    ax3.clear()
    g_ave = (g_sigma + g_pi)/2 
    ax3.plot(nulist,g_ave[frame,:],label = 'avergae modulated line',color = 'b')
    ax3.plot(nulist,g_ave[0,:],label = 'original line',color = 'r')
    ax3.legend()
    ax3.set_ylim([0, 1.2*np.max(g_ave[0,:],axis = None)])
    ax3.set_ylabel('average lineshape fuinction')
    ax3.set_xlabel('frequency [THz]')
    ax3.set_title(f'B-field = {B[frame]:.2f}T')  

if __name__ == "__main__":
    ab_main()
    #ab_Bline()
    