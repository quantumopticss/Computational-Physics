import numpy as np
import matplotlib.pyplot as plt

def fun(t,x,w):
    dxdt = np.array([ x[1],-(w**2*x[0])])
    return dxdt

def ode_test():
    ## users set
    tspan = np.array([0,4])
    x0 = np.array([2,0])
    
    ## operate
    tlist,xlist = ode23i(fun,tspan[0],x0,tspan[1],args = (3,))
    ttlist,xxlist = ode23(fun,tspan[0],x0,tspan[1],args = (3,))
    
    ## figrue
    plt.figure(1)
    plt.plot(tlist,xlist[:,0],label = 'Position23i')
    plt.plot(tlist,xlist[:,1],label = 'Velocity23i')
    plt.plot(ttlist,xxlist[:,0],label = 'Position23')
    plt.plot(ttlist,xxlist[:,1],label = 'Velocity23')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(xlist[:,0],xlist[:,1],label = 'Phase')
    plt.legend()
    plt.show()

    plt.figure(3)
    nt = np.size(tlist)
    ntt = np.size(ttlist)
    nt_list = np.arange(1,nt+1,1)
    ntt_list = np.arange(1,ntt+1,1)
    plt.plot(nt_list,tlist,label = 'implicity')
    plt.plot(ntt_list,ttlist,label = 'explicity')
    plt.legend()
    plt.title('time step')
    plt.show()

def ode23(fun,t_start,initial,t_end,args=(),step_max = 1e-2,TOL = 1e-5):
    ## begin and validity check
    if t_start <= t_end:
        ValueError('t_start should smaller than t_end') 

    t = t_start
    h = step_max*1e-4
    x = initial
    tlist = np.zeros([1,1])
    tlist[0,0] = t
    xlist = initial
    auxi = np.ones_like(x)

    ## operate
    while(t < t_end):
        ## ode23 body
        dxdt_1 = fun(t,x,*args)
        dxdt_2 = fun(t+0.5*h,x+0.5*h*dxdt_1,*args)
        dxdt_3 = fun(t+h,x+h*(-dxdt_1+2*dxdt_2),*args)
        
        delta_2 = 0.5*h*(dxdt_1+dxdt_3)
        delta_3 = h*(dxdt_1 + 4*dxdt_2 + dxdt_3)/6
        ## justify
        e = delta_3 - delta_2
        f = np.max([np.abs(x),auxi],0)
        rato = np.abs(e/f)

        ## results
        Z = np.max(rato,0)/TOL
        if Z < 1:
            x = x + delta_3
            t += h
            tlist = np.vstack((tlist,t))
            xlist = np.vstack((xlist,x))
            h = np.min([(1/Z)**(1/3),step_max])
        else:
            h *= 0.7*((1/Z)**(1/3))
            continue
        
    ## return
    return tlist,xlist

def ode45(fun,t_start,initial,t_end,args=(),step_max = 1e-2,TOL = 1e-5):
    ## begin and validity check
    if t_start <= t_end:
        ValueError('t_start should smaller than t_end') 

    t = t_start
    h = step_max*1e-4
    x = initial
    tlist = np.zeros([1,1])
    tlist[0,0] = t
    xlist = initial
    auxi = np.ones_like(x)

    ## operate
    while(t < t_end):
        ## ode23 body
        dxdt_1 = fun(t,x,*args)
        dxdt_2 = fun(t+0.25*h,x+0.25*h*dxdt_1,*args)
        dxdt_3 = fun(t+0.375*h,x+h*(3*dxdt_1+9*dxdt_2)/32,*args)
        dxdt_4 = fun(t+ 12*h/13,x+h*(1932*dxdt_1 - 7200*dxdt_2 +7296*dxdt_3)/2197,*args)
        dxdt_5 = fun(t+h,x+h*(439*dxdt_1/216 - 8*dxdt_2 + 3680*dxdt_3/513 - 845*dxdt_4/4104),*args)
        dxdt_6 = fun(t+0.5*h,x+h*(-8*dxdt_1/27 + 2*dxdt_2 - 3544*dxdt_3/2565 + 1895*dxdt_4/4104 - 0.275*dxdt_5),*args)
        
        delta_4 = h*(25*dxdt_1/216 + 1408*dxdt_3/2565 + 2197*dxdt_4/4104 - 0.2*dxdt_5)
        delta_5= h*(16*dxdt_1/135 + 6656*dxdt_3/12825 + 28561*dxdt_4/56430 - 0.18*dxdt_5 + 2*dxdt_6/55)
        ## justify
        e = delta_5 - delta_4
        f = np.max([np.abs(x),auxi],0)
        rato = np.abs(e/f)

        ## results
        Z = np.max(rato,0)/TOL
        if Z < 1:
            x = x + delta_5
            t += h
            tlist = np.vstack((tlist,t))
            xlist = np.vstack((xlist,x))
            h = np.min([h*(1/Z)**(1/3),step_max])
        else:
            h *= 0.7*((1/Z)**(1/3))
            continue
        
    ## return
    return tlist,xlist

def ode23i(fun,t_start,initial,t_end,args=(),step_max = 1e-2,TOL = 1e-5):
    ## begin and validity check
    if t_start <= t_end:
        ValueError('t_start should smaller than t_end') 

    t = t_start
    h = step_max*1e-4
    x = initial
    xi = initial
    tlist = np.zeros([1,1])
    tlist[0,0] = t
    xlist = initial
    auxi = np.ones_like(x)

    ## operate
    while(t < t_end):
        ## ode23i body
        # ode2
        dxdt_1 = fun(t,x,*args)
        dxdt_2 = fun(t+h,x+h*dxdt_1,*args)
        
        delta_2 = 0.5*h*(dxdt_1+dxdt_2)
        # ode2i
        xi = x + delta_2 # a estimated x
        f = xi - x - 0.5*h*(dxdt_1 + fun(t+h,xi,*args))
        while(any(f >= 1e-2*TOL)):
            dfdi = 1 - 0.25*1e3*h*(fun(t+h,xi+1e-3,*args) - fun(t+h,xi-1e-3,*args) )
            xi = xi - f/dfdi
            f = xi - x - 0.5*h*(dxdt_1 + fun(t+h,xi,*args))

        delta_i = 0.5*(dxdt_1 + fun(t+h,xi,*args))*h
        ## justify
        e = delta_i - delta_2
        f = np.max([np.abs(xi),auxi],0)
        rato = np.abs(e/f)

        ## results
        Z = np.max(rato,0)/TOL
        if Z < 1:
            t += h
            x = xi
            tlist = np.vstack((tlist,t))
            xlist = np.vstack((xlist,x))
            h = np.min([(1/Z)**(1/3),step_max])
        else:
            h *= 0.7*((1/Z)**(1/3))
            continue
        
    ## return
    return tlist,xlist

if __name__ == "__main__":
    ode_test() 