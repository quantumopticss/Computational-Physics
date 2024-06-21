import numpy as np

def interpolate_lag(xlist,ylist):
    n = np.size(xlist)
    if n != np.size(ylist):
        ValueError('length of xlist should equal to length of ylist')

    def A(i,x):
        j = 0
        A = 1
        while(j < n):
            if j != i:
                A *= (x - xlist[j])/(xlist[i] - xlist[j])
            j += 1
        return A
    
    def fun(x):
        i = 0
        P = 0
        while(i < n):
            P += ylist[i] * A(i,x)
            i += 1
        
        return P

    return fun 


def ip_test():
    xlist = np.linspace(0,3,5)
    tfun = lambda x: np.sin(x)
    ylist = tfun(xlist)

    xxlist = np.linspace(-0.5,3.5,100)
    func = interpolate_lag(xlist,ylist)
    yylist = func(xxlist)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(xxlist,yylist,label = 'interpolate')
    plt.plot(xxlist,tfun(xxlist),label = 'true function')
    plt.scatter(xlist,ylist,label = 'sample point')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ip_test()