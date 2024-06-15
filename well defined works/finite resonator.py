import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
# 2D resonator

def fr_cal(U,X,Y,d,lbd,Rx,Ry):
    PI = np.pi
    result = np.empty_like(Y,dtype = complex)
    n = np.size(result)

    i = 0
    while(i<n):
        Inte = 0
        y = Y[i]
        Inte = np.sum( np.exp(1j*PI*X**2/(lbd*Rx)) * np.exp(-1j*PI*(X-y)**2/(lbd*d))*U ,axis = None) # core
        Inte = Inte*np.exp(1j*PI*y**2/(lbd*Ry))
        result[i] = Inte
        i += 1

    result = (1j)*result*np.exp(-1j*2*PI*d/lbd)/(lbd*d)
    return result

def fr_main():
    ## users set
    # *1[um]
    d = 20.0*1e3 # distance between two mirrors
    R1 = 1.5*d# radius of the first mirror 
    R2 = 1.2*d # radius of the second mirror 
    q = 30000 # longitudinal model
    m = int(2) # transverse model

    a1 = 0.16*1e3 # L1 of the first mirror
    a2 = 0.16*1e3 # L2 of the second mirror
    h = 1 # mesh accuracy
    rounds = 200 # rounds of fb propagation

    ## calculate
    # frequency and lbd
    if R1 == d:
        R1 += 1e-1/np.max([1,R1])
    if R2 == d:
        R2 += 1e-1/np.max([1,R1])

    Z1 = d*(d-R2)/(-2*d+R1+R2)
    Z2 = -d*(d-R1)/(-2*d+R1+R2)
    Z0 = np.sqrt( d*(R1-d)*(R2-d)*(-d+R1+R2) )/(-2*d+R1+R2)
    nu = (q + (m+1)*(np.arctan(Z2/Z0) - np.arctan(Z1/Z0))/np.pi )/(2*d) # *c
    lbd = 1/nu # c_const/nu
    W0 = np.sqrt(lbd*Z0/np.pi)

    # mesh
    X1 = np.arange(-a1,a1+h,h)
    X2 = np.arange(-a2,a2+h,h)

    # initial
    # make initial condition # H2(x) = 4*x**2 -2 
    U1 = np.ones_like(X1,dtype = complex) 
    U2 = np.empty_like(X2,dtype = complex)
    # calculate
    i = 0
    while(i<rounds):
        U2 = fr_cal(U1,X1,X2,d,lbd,R1,R2) # propagation forth
        U2 = U2/np.sum(np.abs(U2))
        U1 = fr_cal(U2,X2,X1,d,lbd,R2,R1) # propagation back
        i += 1

    ## figure
    # normalize
    I2 = np.abs(U2)**2
    I1 = np.abs(U1)**2
    I2 = I2/np.sum(I2)
    I1 = I1/np.sum(I1)

    # parameters
    W2 = W0*np.sqrt(1+(Z2/Z0)**2)
    W1 = W0*np.sqrt(1+(Z1/Z0)**2)

    I1_c = eval_hermite(m,np.sqrt(2)*X1/W1)*np.exp(-(X1/W1)**2)
    I2_c = eval_hermite(m,np.sqrt(2)*X2/W2)*np.exp(-(X2/W2)**2)
    I1_c = np.abs(I1_c)**2; I1_c = I1_c/np.sum(I1_c)
    I2_c = np.abs(I2_c)**2; I2_c = I2_c/np.sum(I2_c)
    
    plt.figure(1)
    # figure 1
    plt.subplot(1,2,1)
    plt.plot(X1,I1,label = 'simulate')
    plt.plot(X1,I1_c,label = 'theoretic')
    plt.title('first mirror')

    # figure 2
    plt.subplot(1,2,2)
    plt.plot(X2,I2,label = 'simulate')
    plt.plot(X2,I2_c,label = 'theoretic')
    plt.title('second mirror')

    plt.suptitle('intensity pattern on the mirror')
    plt.show()

if __name__ == "__main__":
    fr_main()
 