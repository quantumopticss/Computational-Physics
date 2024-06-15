import numpy as np
import matplotlib.pyplot as plt

def sse_funV(x,v1,v2):
    # periodic = pi
    f = np.sign(np.cos(2*np.pi*x))*(v2-v1)/2 + (v2+v1)/2
    return f

def sse_funM(x):
    return 1.0 + 0.5*np.sign(np.cos(2*np.pi*x+np.pi/4))

def sse_T(funM,xlist,args = ()):
    n = np.size(xlist)
    T = np.zeros([n,n])
    T[0,[0,1]] = np.array([-2,1])/funM(xlist[0],*args)
    T[n-1,[n-2,n-1]] = np.array([1,-2])/funM(xlist[n-1],*args)
    
    i = 1
    while(i<n-1):
        T[i,[i-1,i,i+1]] = np.array([1,-2,1])/funM(xlist[i],*args)
        i += 1 
    
    return T

def sse_V(funV,xlist,args = ()):
    n = np.size(xlist)
    V = np.zeros([n,n])
    i = 0
    while(i<n):
        V[i,i] = funV(xlist[i],*args)
        i += 1
    return V

def sse_main():
    ## users set
    h = 0.01 # mesh size
    L = 2   # simulation area
    v1 = 100
    v2 = 200

    # physics consts
    hbar = 1

    ## operate
    # potential energy
    N = int(np.ceil(L/h)) # dimension
    xlist = np.arange(0,N-1+1,1)*h
    V = sse_V(sse_funV,xlist,args = (v1,v2,))
    
    # kinetic energy
    T = sse_T(sse_funM,xlist,args = ())
    T = T*(-1)*hbar**2/(2*h**2)
    
    # Hamiltonion
    H = T + V # H\T = E\T
    
    eigval,eigvec = np.linalg.eigh(H)

    ## figure
    plt.figure(1)
    i = 20
    while (i<180):
        plt.plot(np.real(eigval),'r')
        plt.plot(np.imag(eigval),'b')
        i += 20
    plt.legend()
    plt.show()



if __name__ == "__main__":
    sse_main()