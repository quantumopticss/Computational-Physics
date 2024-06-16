import numpy as np
### 1D integral
def integral_subcalculate(fun,x_start,x_end,args = ()):
    # calculate a small integrate over xspan

    Xlist4 = np.linspace(0,1,5,endpoint = True)*(x_end - x_start) + x_start
    i4 = fun(Xlist4,*args)
    Co_4 = np.array([7,32,12,32,7])

    I = (np.sum(Co_4*i4,axis = None))*(x_end - x_start)/90
    return I

def integral_operate(fun,xspan,args = (),TOL = 1e-6):
    N = 4
    int_space = np.linspace(xspan[0],xspan[1],N,endpoint = True)
    result = 0
    j = 0
    n = 4
    while(j<(N-1)): ## integral in int_space
        n = np.max([N//2,n//2])
        erf = 1
        result_0 = integral_subcalculate(fun,int_space[j],int_space[j+1],args)
        while(erf >= TOL): ## self_adjust integral in span: { int_space[j],int_space[j+1] }
            sub_space = np.linspace(int_space[j],int_space[j+1],n,endpoint = True); 
            result_1 = 0
            k = 0
            while(k<(n-1)): ## integral in sub_space
                result_1 += integral_subcalculate(fun,sub_space[k],sub_space[k+1],args)
                k += 1
            n *= 2

            erf = np.abs(result_1 - result_0)/np.max([1,np.abs(result_1)])
            result_0 = result_1 

        result += result_0
        j += 1
    
    return result

def integral(fun,xspan,args = (),TOL = 1e-6):
    """ numerical integral
        support complex function: np.sqrt(-1+0j) = 1j
        fun:  function to integrate,
        xpans: integral span, 
        TOL: value to control numerical error
    """
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
            result = integral_operate(fun,[x0,x0+step],args,TOL)
            e = 1
            x0 += step
            while(e>=TOL):
                add = integral_operate(fun,[x0,x0+step],args,TOL)
                result += add
                x0 += step
                e = np.abs( add/np.max([np.abs(result),1])  )

        case [-1,0]:
            step = 10
            result = integral_operate(fun,[x1-step,x1],args,TOL)
            e = 1
            x1 -= step
            while(e>=TOL):
                add = integral_operate(fun,[x1-step,x1],args,TOL)
                result += add
                x1 -= step
                e = np.abs( add/np.max([np.abs(result),1])  )

        case [-1,1]:
            # [0,1]
            x0 = 0
            step = 10
            result_f = integral_operate(fun,[x0,x0+step],args,TOL)
            e = 1
            x0 += step
            while(e>=TOL):
                add = integral_operate(fun,[x0,x0+step],args,TOL)
                result_f += add
                x0 += step
                e = np.abs( add/np.max([np.abs(result_f),1])  )

            #[-1,0]
            x1 = 0
            result_b = integral_operate(fun,[x1-step,x1],args,TOL)
            e = 1
            x1 -= step
            while(e>=TOL):
                add = integral_operate(fun,[x1-step,x1],args,TOL)
                result_b += add
                x1 -= step
                e = np.abs( add/np.max([np.abs(result_b),1])  )

            result = result_f + result_b

        case _:
            result = integral_operate(fun,[x0,x1],args,TOL)

    return (result*sgn)

### 2D integral
def integral2D_subcalculate(yxfun,funy_down,funy_up,x_start,x_end,args=()):
    xlist = np.linspace(x_start,x_end,5,endpoint = True)
    Co_4 = np.array([7,32,12,32,7])
    result = 0
    i = 0
    while(i<5):
        int_y = integral(yxfun,[funy_down(xlist[i]),funy_up(xlist[i])],(xlist[i],*args))
        result += int_y*Co_4[i]
        i += 1

    return (result*(x_end - x_start)/90)

def integral2D_operate(yxfun,funy_down,funy_up,xspan,args=(),TOL = 1e-6):
    N = 4
    int_xspace = np.linspace(xspan[0],xspan[1],N,endpoint = True)
    result = 0
    j = 0
    n = 4
    while(j<(N-1)): ## integral in int_space
        n = np.max([N//2,n//2])
        erf = 1
        result_0 = integral2D_subcalculate(yxfun,funy_down,funy_up,int_xspace[j],int_xspace[j+1],args)
        while(erf >= TOL): ## self_adjust integral in span: { int_space[j],int_space[j+1] }
            sub_space = np.linspace(int_xspace[j],int_xspace[j+1],n,endpoint = True); 
            result_1 = 0
            k = 0
            while(k<(n-1)): ## integral in sub_space
                result_1 += integral2D_subcalculate(yxfun,funy_down,funy_up,sub_space[k],sub_space[k+1],args)
                k += 1
            n *= 2

            erf = np.abs(result_1 - result_0)/np.max([1,np.abs(result_1)])
            result_0 = result_1 

        result += result_0
        j += 1
    
    return result

def integral2D(yxfun,funy_down,funy_up,xspan,args = (),TOL = 1e-6):
    """ 2D numerical integral
        support complex function: np.sqrt(-1+0j) = 1j
        yxfun: function to integrate, but you should use function of this form: func(y,x,*args)
        xpans: integral span of x-axis
        funy: funtion of y lines
        TOL: value to control numerical error
    """
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
            1

        case [-1,0]:
            1

        case [-1,1]:
            1

        case _:
            result = integral2D_operate(yxfun,funy_down,funy_up,xspan,args,TOL)

    return (result*sgn)

### 1D discrete integral
def int_integral(ylist,xlist,order = 4,TOL = 1e-6):
    """ discrete numerical integral, first do interpolation, and then numerical integrate
        ylist: discrete sampling date from sensor you want to integrate over xlist 
        xlist: span over which you want to do integration about ylist
        order: value to control interpolate error 1 <= order <= 5
        TOL: value to control integrate error
    """
    from scipy.interpolate import UnivariateSpline
    spline = UnivariateSpline(xlist,ylist,k = order)

    fun = lambda x: spline(x)
    result = integral(fun,[xlist[0],xlist[-1]],(),TOL)
    return result,fun

# testing functions  
def yxfun(y,x):
    f = y**2*x**2
    return f

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #print(np.sqrt(-1+0j)) --> 1j

    funy_up = lambda x: 1 
    funy_down = lambda x: 0
    xspan = [0,1]
    S = integral2D(yxfun,funy_down,funy_up,xspan,args = (),TOL = 1e-4)
    print(S)


    