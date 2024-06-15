import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def rd_test():
    ## test
    N = 6000
    pspan = np.array([0,4]);ran = pspan[1] - pspan[0]
    result = np.empty([2,N])
    i = 0
    fun = lambda x: np.exp(-0.3*x**2)
    A = quad(fun,pspan[0],pspan[1])[0]
    bi = 100

    while(i<N):
        result[0,i] = rd_dis_choose(fun,pspan,TOL = 0.001)
        result[1,i] = rd_dis_fun(fun,pspan,TOL = 0.001)
        i += 1
    ## figure
    xlist = np.linspace(pspan[0],pspan[1],200,endpoint = True)

    plt.subplot(1,2,1)
    plt.plot(xlist,fun(xlist)*ran*N/bi/A)
    plt.hist(result[0,:], bins = bi)
    plt.title('distribution density -- choose')

    plt.subplot(1,2,2)
    plt.plot(xlist,fun(xlist)*ran*N/bi/A)
    plt.hist(result[1,:], bins = bi)
    plt.title('distribution sensity -- fun')

    plt.show()

def rd_dis_fun(fun,pspan,args = (),TOL = 1e-4):
    if pspan[0] >= pspan[1]:
        ValueError('pspan error')
    
    ## normalize fun

    # case _:
    int_list = np.linspace(pspan[0],pspan[1],100,endpoint = True) # only not [-infty,infty]
    Alist = np.zeros_like(int_list)
    i = 1
    while (i<100):
        Alist[i] = quad(fun,pspan[0],int_list[i],args)[0]
        i += 1

    A = Alist[-1]
    Alist = Alist/A
    p = np.random.rand()

    n = np.sum((Alist<p),axis = None) - 1 # location of P in (int_list[n],ini_list[n+1] )
    
    A_down = Alist[n] # the nearest A
    x0 = int_list[n]

    e = 1
    delta = A*(p - A_down)/fun(x0,*args)
    h = 1 # step factor
    while (e>=TOL):
        e = A*(p - A_down) - quad(fun,x0,x0+delta,args)[0]
        delta += h*e/fun(x0+delta,*args)
        h *= (TOL/np.abs(e))
    
    return (x0+delta)

def rd_dis_choose(fun,pspan,args = (),TOL = 1e-4):
    if pspan[0] >= pspan[1]:
        ValueError('pspan error')
    
    ## normalize fun

    # case _:
    sta_list = np.arange(pspan[0],pspan[1],TOL) + TOL/2 # only not [-infty,infty]
    value_list = fun(sta_list,*args)
    value_list = value_list/np.max(value_list)
    sta_list = sta_list - TOL/2

    while (True):
        p = pspan[0] + (pspan[1] - pspan[0])*np.random.rand()
        n = np.sum((p>sta_list))-1 # p in the location of sta_list

        choose_p = np.random.rand()
        if (choose_p <= value_list[n]):
            break
        
    return p

def rand_distributed(fun,pspan,args = (),TOL = 1e-4,method = "function"):
    match method:
        case "function":
            p = rd_dis_fun(fun,pspan,args,TOL)

        case "choose":
            p = rd_dis_choose(fun,pspan,args,TOL)

        case _:
            ValueError("method error, only support 'function' and 'choose'")
    
    return p

if __name__ == "__main__":
    rd_test()
    
    #a = np.array([1,2,3])
    #f = lambda x: x
    #print(f(a))
    #print(quad(f,0,1)[0]) -> 0.5