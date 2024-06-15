import numpy as np
import matplotlib.pyplot as plt

#def fun(x,a):
#    f = np.exp(-a*x**2)
#    return f
## lambda: f
def integral_calculate45(fun,xspan, args=() ,h_step = 1e-2,TOL = 1e-4):
    x0 = xspan[0]
    x1 = xspan[1]

    h = h_step
    result = 0
    while(x0<x1):
        Xlist41 = np.linspace(0,1,5,endpoint = True)*h + x0
        Xlist42 = np.linspace(0,1,9,endpoint = True)*h + x0
        I41 = fun(Xlist41,*args)
        I42 = fun(Xlist42,*args)
        Co_41 = np.array([7,32,12,32,7])/90
        Co_42 = np.array([7,32,12,32,14,32,12,32,7])/180

        I1 = np.sum(Co_41*I41,axis = None)
        I2 = np.sum(Co_42*I42,axis = None)
        
        erf = np.abs(I2 - I1)
        e = (erf/4*np.max([np.abs(I2),1]))*(1/(h_step))**2
        print(x0)
        if e < TOL:
            x0 += h
            result += I2*h
            h = h_step
        else:
            h *= 0.5*(TOL/e)**0.3
            if h < 1e-9:
                ValueError (f'function singular at x = {x0}')

    return result

def integral(fun,xspan,args = (),h_step = 1e-2,TOL = 1e-4):
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
            result = integral_calculate45(fun,[x0,x0+step],args,TOL)
            e = 1
            x0 += step
            while(e>=TOL):
                add = integral_calculate45(fun,[x0,x0+step],args,TOL)
                result += add
                x0 += step
                e = np.abs( add/np.max([np.abs(result),1])  )

        case [-1,0]:
            step = 10
            result = integral_calculate45(fun,[x1-step,x1],args,TOL)
            e = 1
            x1 -= step
            while(e>=TOL):
                add = integral_calculate45(fun,[x1-step,x1],args,TOL)
                result += add
                x1 -= step
                e = np.abs( add/np.max([np.abs(result),1])  )

        case [-1,1]:
            # [0,1]
            x0 = 0
            step = 10
            result_f = integral_calculate45(fun,[x0,x0+step],args,TOL)
            e = 1
            x0 += step
            while(e>=TOL):
                add = integral_calculate45(fun,[x0,x0+step],args,TOL)
                result_f += add
                x0 += step
                e = np.abs( add/np.max([np.abs(result_f),1])  )

            #[-1,0]
            x1 = 0
            result_b = integral_calculate45(fun,[x1-step,x1],args,TOL)
            e = 1
            x1 -= step
            while(e>=TOL):
                add = integral_calculate45(fun,[x1-step,x1],args,TOL)
                result_b += add
                x1 -= step
                e = np.abs( add/np.max([np.abs(result_b),1])  )

            result = result_f + result_b

        case _:
            result  = integral_calculate45(fun,[x0,x1],args,TOL)

    ## return 
    return (result*sgn)  

def integral2D(fun,xspan,yfun_down,yfun_up,args = (), h_step = 1e-3,TOL = 1e-3):
    ## iint f(x,y) d2S = int dx int_{y1}^{y2} dy
    ## fun(y,x,args)
    if xspan[1] == xspan[0]:
        return 0
    
    ## x1 > x0 
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
    
    ## 2D integral
    match process:
        case [0,1]:
            step = 10
            result = integral_2Dcalculate(fun,[x0,x0+step],yfun_down,yfun_up,args,TOL)
            e = 1
            x0 += step
            while(e>=TOL):
                add = integral_2Dcalculate(fun,[x0,x0+step],yfun_down,yfun_up,args,TOL)
                result += add
                x0 += step
                e = np.abs( add/np.max([np.abs(result),1])  )

        case [-1,0]:
            step = 10
            result = integral_2Dcalculate(fun,[x1-step,x1],yfun_down,yfun_up,args,TOL)
            e = 1
            x1 -= step
            while(e>=TOL):
                add = integral_2Dcalculate(fun,[x1-step,x1],yfun_down,yfun_up,args,TOL)
                result += add
                x1 -= step
                e = np.abs( add/np.max([np.abs(result),1])  )

        case [-1,1]:
            # [0,1]
            x0 = 0
            step = 10
            result_f = integral_2Dcalculate(fun,[x0,x0+step],yfun_down,yfun_up,args,TOL)
            e = 1
            x0 += step
            while(e>=TOL):
                add = integral_2Dcalculate(fun,[x0,x0+step],yfun_down,yfun_up,args,TOL)
                result_f += add
                x0 += step
                e = np.abs( add/np.max([np.abs(result_f),1])  )

            #[-1,0]
            x1 = 0
            result_b = integral_2Dcalculate(fun,[x1-step,x1],yfun_down,yfun_up,args,TOL)
            e = 1
            x1 -= step
            while(e>=TOL):
                add = integral_2Dcalculate(fun,[x1-step,x1],yfun_down,yfun_up,args,TOL)
                result_b += add
                x1 -= step
                e = np.abs( add/np.max([np.abs(result_b),1])  )

            result = result_f + result_b

        case _:
            result = integral_2Dcalculate(fun,xspan,yfun_down,yfun_up,args,TOL)

    ## return 
    return (result*sgn)  

def integral_2Dcalculate(fun,xspan,yfun_down,yfun_up,args,h_step = 1e-3,TOL = 1e-4):
    # x0 < x1
    x0 = xspan[0]
    x1 = xspan[1]
    result = 0.0
    h = h_step

    while (x0<x1):
        I0 = integral(fun,[yfun_down(x0),yfun_up(x0)],(x0,*args),TOL)
        Ie = integral(fun,[yfun_down(x0+h),yfun_up(x0+h)],(x0+h,*args),TOL)

        Ia = integral(fun,[yfun_down(x0+0.25*h),yfun_up(x0+0.25*h)],(x0+0.25*h,*args),TOL)
        Ib = integral(fun,[yfun_down(x0+0.5*h),yfun_up(x0+0.5*h)],(x0+0.5*h,*args),TOL)
        Ic = integral(fun,[yfun_down(x0+0.75*h),yfun_up(x0+0.75*h)],(x0+0.75*h,*args),TOL)

        I1 = (7*I0 + 32*Ib + 12*Ia + 32*Ic)/90

        I_a = integral(fun,[yfun_down(x0+0.125*h),yfun_up(x0+0.125*h)],(x0+0.125*h,*args),TOL)
        I_b = integral(fun,[yfun_down(x0+0.375*h),yfun_up(x0+0.375*h)],(x0+0.375*h,*args),TOL)
        I_c = integral(fun,[yfun_down(x0+0.625*h),yfun_up(x0+0.625*h)],(x0+0.625*h,*args),TOL)
        I_d = integral(fun,[yfun_down(x0+0.875*h),yfun_up(x0+0.875*h)],(x0+0.875*h,*args),TOL)

        I2 = (7*I0 + 32*I_a + 12*Ia + 32*I_b + 14*Ib + 32*I_c + 12*Ic + 32*I_d + 7*Ie)/180
        erf = np.abs(I2-I1)
        e = (erf/4*np.max([np.abs(I2),1]))*(1/h_step)**2

        if e < TOL:
            x0 += h
            result += I2*h
            h = h_step
        else:
            h *= 0.5*(TOL/e)**0.3
            if h < 1e-9:
                ValueError (f'function singular at x = {x0}')

    return result 

def d_integral(ylist,xlist,)

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

def fun(x):
    f = np.exp(x)
    return f

if __name__ == "__main__":
    #test()
    #print(np.sqrt(-1+0j)) --> 1j

    S = integral(fun,[3,6],args = (),h_step = 0.1,TOL = 0.01)
    print(S)
    print(np.exp(6) - np.exp(3))
    
    #yfun1 = lambda x: x
    #yfun0 = lambda x: 0
    #S = integral2D(fun,[0,1],yfun0,yfun1,args = (),h_step = 0.01,TOL = 1e-3)
    #print(S)