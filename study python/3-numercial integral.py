import numpy as np
import matplotlib.pyplot as plt

## rho(x) = rho0*exp(-x^2)

def ni_integral(x0,y0,source,X,h):
    ## int source/(sqrt( (x-x0)^2+y^2  )) dx
    
    result = (source[1] + source[2] + source[-1] + source[-2])*h/2

    i = 2
    return result

def ni_main():
    ## users set
    h = 0.01
    c = 5;r = 5 # size of area

    ## operate
    nx = int(r/h); ny = int(c/h)
    T = np.zeros([nx+2,ny+2])
    
    X = np.arange(1,nx+1,1);Y = np.arange(1,ny+1,1) # digital location 1:1:nx
    source = np.exp(-(h*X - h*int(nx/2))**2)

    for x in X:
        if x == 0:
            continue
        for y in Y:
            T[x,y] = ni_integral(x,y,source,h)

    ## figure
    plt.figure(1)
    plt.imshow(T)
    plt.title("T")
    plt.show()


    
if __name__ == "__main__":
    ni_main()