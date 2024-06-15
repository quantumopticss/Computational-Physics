import numpy as np
import matplotlib.pyplot as plt

## rho(x) = rho0*exp(-x^2)

def ef_main():
    ## users set
    h = 0.1  # discret size
    r = 5;c=5 # total size

    nx = int(r/h);ny = int(c/h)
    mesh = np.zeros([nx+2,ny+2])

    X = np.linspace(0,c,nx+2,endpoint = True) - r/2
    Y = np.zeros([1,ny+2])
    #source[] = 

    ## figrue

    
if __name__ == "__main__":
    ef_main()