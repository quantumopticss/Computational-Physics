import numpy as np

def integral_subcalculate(fun,x_start,x_end,args = ()):
    # calculate a small integrate over xspan

    Xlist4 = np.linspace(0,1,5,endpoint = True)*(x_end - x_start) + x_start
    i4 = fun(Xlist4,*args)
    Co_4 = np.array([7,32,12,32,7])

    I = (np.sum(Co_4*i4,axis = None))*(x_end - x_start)/90
    return I

def integral_operate(fun,xspan,args = (),TOL = 1e-8):
    N = 8
    int_space = np.linspace(xspan[0],xspan[1],N,endpoint = True)
    result = 0
    j = 0
    n = 4
    while(j<(N-1)): ## integral in int_space
        n = np.max([4,n//2])
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

def integral(fun,xspan,args = (),TOL = 1e-8):
    """numerical integral
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

def int_integral(ylist,xlist,order = 3):
    1

def fun(x,k,l):
    f = 1
    return f

if __name__ == "__main__":
    #print(np.sqrt(-1+0j)) --> 1j
    k = 3; l = 4
    S = integral(fun,[0,1],args = (k,l,),TOL = 1e-3)

    