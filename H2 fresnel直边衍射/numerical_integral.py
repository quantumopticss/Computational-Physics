## *args 是打包和解包
# 在传递args的时候不加*
# 例如integral接受调用处的 args = (a,b,...)
# 传递给 integral_calculate(...,args,...)
# integral_calculate 调用 func 需要解包，fun(...,*args,...)

import numpy as np
import matplotlib.pyplot as plt

#def fun(x,a):
#    f = np.exp(-a*x**2)
#    return f
## lambda: f
def fun(x,a,b):
    f = b*np.exp(-a*x**2)
    return f

def integral_calculate(fun,xspan, args=() ,h_step = 1e-3,TOL = 1e-4):
    x0 = xspan[0]
    x1 = xspan[1]

    h = h_step/10 
    result = 0
    while(x0<x1):
        I1 = h*(fun(x0,*args) +4*fun(x0+h/2,*args) +fun(x0+h,*args) )/6 # 
        I2 = h*( fun(x0,*args) + 3*fun(x0+h/3,*args) + 3*fun(x0+2*h/3,*args) + fun(x0+h,*args) )/8
        
        erf = np.abs(I2 - I1)
        e = erf/np.max([np.abs(I2),1])

        if e <= TOL:
            x0 += h
            result += I2
            h = np.min([h*(TOL/(e+TOL**2))**0.3,h_step]) 
        else:
            h *= 0.8*(TOL/e)**0.5
            if h < 1e-9:
                ValueError (f'function singular at x = {x0}')

    return result

def integral(fun,xspan,args = (),h_step = 1e-3,TOL = 1e-4):
    if xspan[1] == xspan[0]:
        return 0
    
    if xspan[1] > xspan[0]:
        sgn = 1
        x0 = xspan[0]
        x1 = xspan[1]
    else:
        sgn = -1
        x0 = xspan[1]
        x1 = xspan[0]
    
    process = [0,0]
    if x1 == np.infty:
        process[1] = 1
    if x0 == -np.infty:
        process[0] = -1
    
    ## integral
    match process:
        case [0,1]:
            step = 10
            result = integral_calculate(fun,[x0,x0+step],args,TOL)
            e = 1
            x0 += step
            while(e>=TOL):
                add = integral_calculate(fun,[x0,x0+step],args,TOL)
                result += add
                x0 += step
                e = np.abs( add/np.max([np.abs(result),1])  )

        case [-1,0]:
            step = 10
            result = integral_calculate(fun,[x1-step,x1],args,TOL)
            e = 1
            x1 -= step
            while(e>=TOL):
                add = integral_calculate(fun,[x1-step,x1],args,TOL)
                result += add
                x1 -= step
                e = np.abs( add/np.max([np.abs(result),1])  )

        case [-1,1]:
            # [0,1]
            x0 = 0
            step = 10
            result_f = integral_calculate(fun,[x0,x0+step],args,TOL)
            e = 1
            x0 += step
            while(e>=TOL):
                add = integral_calculate(fun,[x0,x0+step],args,TOL)
                result_f += add
                x0 += step
                e = np.abs( add/np.max([np.abs(result_f),1])  )

            #[-1,0]
            x1 = 0
            result_b = integral_calculate(fun,[x1-step,x1],args,TOL)
            e = 1
            x1 -= step
            while(e>=TOL):
                add = integral_calculate(fun,[x1-step,x1],args,TOL)
                result_b += add
                x1 -= step
                e = np.abs( add/np.max([np.abs(result_b),1])  )

            result = result_f + result_b

        case _:
            result  = integral_calculate(fun,[x0,x1],args,TOL)

    ## return 
    return (result*sgn)  

def test():
    a = 0.5 # users parametera
    xspan = np.array([-10,10]) # integral span

    #fun = lambda x,a: x**4*np.exp(-a*x**2)    
    xlist = np.linspace(xspan[0],xspan[1],5)
    xlist[-1] = np.infty
    xlist[0] = -np.infty
    relist = np.empty_like(xlist)
    i = 0
    while (i<np.size(xlist)):
        result = integral(fun,[xlist[0],xlist[i]],args = (a,1,))
        relist[i] = result
        i += 1

    ## figure
    plt.figure(1)
    plt.scatter(xlist,relist,label = 'area integral')
    plt.plot(xlist,np.sqrt(np.pi/(a))*np.ones_like(xlist),'r',label = 'final line')
    plt.legend()
    plt.title('self-adjusting numerical integral')
    plt.show()

if __name__ == "__main__":
    test()
    #print(np.sqrt(-1+0j)) --> 1j