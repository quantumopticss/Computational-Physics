import numpy as np
import odesolver as ode
import matplotlib.pyplot as plt

def no_modulate(t,f,A,circle):
    theta = (2*np.pi*f*t)%(2*np.pi)
    theta /= (2*np.pi)

    if circle == 'Negative':
        if theta < 0.25:
            return (1+A), (1+A)
        elif theta < 0.5:
            return (1-A), (1+A)
        elif theta < 0.75:
            return (1-A), (1-A)
        else: 
            return (1+A), (1-A)
    
    if circle == 'Positive':
        if theta < 0.25:
            return (1+A), (1+A)
        elif theta < 0.5:
            return (1+A), (1-A)
        elif theta < 0.75:
            return (1-A), (1-A)
        else: 
            return (1-A), (1+A)

def fun(t,x,w1,w2,gamma1,gamma2,m1,m2,f,A,circle):
    # dxdt = [x1',x1'',x2',x2'']
    # x = [x1,x1',x2,x2']

    fa, fw = no_modulate(t,f,A,circle)

    F = 0.5*np.sqrt(m1*m2*w1*w2)*np.abs(gamma1 - gamma2)*fa
    w_mod = np.abs(w2 - w1)*fw

    a1 = F*np.cos(w_mod*t)*(x[0] - x[2])/m1 - gamma1*x[1] - w1**2*x[0]
    a2 = -F*np.cos(w_mod*t)*(x[0] - x[2])/m2 - gamma2*x[3] - w2**2*x[2]
    dxdt = np.array([x[1],a1,x[3],a2])

    return dxdt

def no_main():
    ## parameters for nonhermite 
    w1 = 280
    w2 = 300
    gamma1 = 0.01
    gamma2 = 0.001
    f = 20
    C_list = ['Positive','Negative']
    circle = C_list[0]
    m1 = 1
    m2 = 0.7
    A = 0.25

    ## parameters for oscillator
    x0 = [1,0,0,0]
    tlist,xlist = ode.ode45(fun,0,x0,4,args = (w1,w2,gamma1,gamma2,m1,m2,f,A,circle),step_max = 0.005,TOL = 1e-5)

    ## figrue

    x1 = xlist[:,0]
    x2 = xlist[:,2]
    v1 = xlist[:,1]
    v2 = xlist[:,3]

    E1 = 0.5*m1*(v1**2 + w1**2*x1**2)
    E2 = 0.5*m2*(v2**2 + w2**2*x2**2)

    plt.figure(1)
    plt.plot(tlist,E1,label = 'E1')
    plt.plot(tlist,E2,label = 'E2')
    plt.plot(tlist,(E1+E2),label = 'total E')
    plt.legend()
    plt.title(circle + 'circle')
    plt.xlabel('time/s')
    plt.ylabel('amplitude')
    plt.show()

if __name__ == "__main__":
    no_main()