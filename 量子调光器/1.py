
nu_down = 508.480 # THz 
nu_up = 508.998 # THz 

import numpy as np
nulist = np.linspace(nu_down,nu_up,200)

import matplotlib.pyplot as plt
T_list = np.ones_like(nulist)

T1_list = np.copy(T_list)
T2_list = np.copy(T_list)

ab_nu_down_1 = 508.7
ab_nu_up_1 = 508.8
T1_list[(nulist <= ab_nu_up_1)*(nulist >= ab_nu_down_1)] = 0.25

def update(frame,ax,nulist,T1_list,T2_list):
    f_ab = np.sum((np.ones_like(T1_list) - T2_list[frame,:]) ,axis = None)*0.01
    ax.clear()
    ax.plot(nulist,T2_list[frame,:],label = f'modulated transmittance, absorb = {f_ab:.2f}')
    ax.plot(nulist,T1_list,label = f'original transmittance, absorb = {np.sum((np.ones_like(T1_list) - T1_list) ,axis = None)*0.01:.2f}')
    ax.legend()
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel('lineshape fuinction')
    ax.set_xlabel('frequency/THz')

    
klist = np.arange(0.01,0.15,0.005)
T2_list = np.zeros([len(klist),len(T1_list)])
for k in range(len(klist)):
    T0 = np.ones_like(T1_list)
    ab_nu_down_2a = ab_nu_down_1 - klist[k]
    ab_nu_up_2a = ab_nu_up_1 - klist[k]

    ab_nu_down_2b = ab_nu_down_1 + klist[k]
    ab_nu_up_2b = ab_nu_up_1 + klist[k]

    T0[(nulist <= ab_nu_up_2a)*(nulist >= ab_nu_down_2a)] = 0.5*T0[(nulist <= ab_nu_up_2a)*(nulist >= ab_nu_down_2a)]
    T0[(nulist <= ab_nu_up_2b)*(nulist >= ab_nu_down_2b)] = 0.5*T0[(nulist <= ab_nu_up_2b)*(nulist >= ab_nu_down_2b)]
    
    T2_list[k,:] = T0

from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()
anim = FuncAnimation(fig, update, frames = len(klist),interval=200,fargs = (ax,nulist,T1_list,T2_list))
anim.save('modulated transmittance.gif', writer='imagemagick')

# ab_nu_down_2a = ab_nu_down_1 - 0.15
# ab_nu_up_2a = ab_nu_up_1 - 0.15

# ab_nu_down_2b = ab_nu_down_1 + 0.15
# ab_nu_up_2b = ab_nu_up_1 + 0.15

# T2_list = np.ones_like(T1_list)
# T2_list[(nulist <= ab_nu_up_2a)*(nulist >= ab_nu_down_2a)] = 0.5*T2_list[(nulist <= ab_nu_up_2a)*(nulist >= ab_nu_down_2a)]
# T2_list[(nulist <= ab_nu_up_2b)*(nulist >= ab_nu_down_2b)] = 0.5*T2_list[(nulist <= ab_nu_up_2b)*(nulist >= ab_nu_down_2b)]
# plt.figure(1)
# plt.plot(nulist,T1_list)
# plt.xlabel('frequency/THz')
# plt.ylim((-0.1,1.1))
# plt.title(f'transmittance, absorb = {np.sum((1 - T1_list),axis = None)*0.01:.2f}')

# plt.figure(2)
# plt.plot(nulist,T2_list)
# plt.xlabel('frequency/THz')
# plt.title(f'transmittance, absorb = {np.sum((1 - T2_list),axis = None)*0.01:.2f}')
# plt.ylim((-0.1,1.1))

# plt.show()


