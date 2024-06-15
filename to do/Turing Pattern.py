import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45

# du/dt = D_u * nabla^2 *u - u*v^2 + F(1-u) 
# dv/dt = D_v * nabla^2 *v + u*V^2 - (F+K)*v

# a
# a
# a
#   b b b b b b
def tp_main():
    ## users set
    h = 0.01 # accuracy
    a = 3; b = 5 # germetry

    ## initialize 
    nx = int(a/h); ny = int(b/h)
    Cu = np.zeros([nx+2,ny+2]); Cv = np.zeros_like(Cu)

    xlist = np.arange(0,nx+2); ylist = np.arange(0,ny+2)  # 0 -- nx+1; 0 -- ny+2
    


def tp_operator():
    #
    1



if __name__ == "__main__":
    tp_main()