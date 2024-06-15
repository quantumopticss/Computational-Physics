## 自回避高斯链
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def crw_main():
    ## users set
    seed = 37
    steps = 120 # make sure that steps is smaller than 2*L
    a = 0.5 # judge radius while step h = 1
    rounds = 2000 # rounds to average
    
    ## operate
    np.random.seed(seed)         # set random seed
    rho_list = np.zeros([1,rounds])
    rds = 0
    while(rds<rounds):
        rho_list[0,rds] = crw_operate_2(steps,a,"distance")
        rds += 1

    xlist,ylist = crw_operate_2(steps,a,"array")

    ## figure
    plt.figure(1)
    hist,edge = np.histogram(rho_list,bins = 15)
    hist = hist/rounds
    dist = 0.5*(edge[0:-1] + edge[1:])
    prob = hist/(2*np.pi*dist)
    plt.plot(dist,prob)
    plt.xlabel("Endpoint Distance R")
    plt.ylabel("Frequencies")
    plt.title(f"disperse density step = {steps} with average time = {rounds} @ overleaf radius a = {a}")
    plt.show()

    plt.figure(2)
    plt.plot(xlist,ylist);plt.scatter(xlist,ylist,c = 'y')
    plt.scatter(xlist[0],ylist[0],c = 'r');plt.scatter(xlist[-1],ylist[-1],c = 'r')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("figure of constrained random walk")
    plt.show()

def crw_operate_1(steps,a,target="array"): # constrained random walk with method1
    ## operate
    PI = np.pi
    R = np.array([[0,0]])
    R0 = np.array([[0,0]])
    r = np.zeros([1,2]) # [[x1,y1],[x2,y2].[x3,y3],...]    # where we store position data
    
    i = 0 # step counter
    a2 = a**2 # turn a to a^2 
    while(i<steps):
        rec = 0 # remake counter
        while(1): 
            rec += 1

            p = 2*PI*np.random.rand()       # random angle
            R0 = R + np.array([[np.cos(p),np.sin(p)]])  # random walk

            ## judge whether points overleaf
            dist = (r[:,0] - R0[0,0])**2 + (r[:,1]-R0[0,1])**2
            dist = (dist < a2)
            f = np.sum(dist,axis = 0)

            # if there is no overleaf we will receive this step and break to the next
            # else remake and rec += 1
            if f == 0:
                R = R0
                r = np.vstack((r,R))
                i += 1
                break

            if rec > 1.5*(i+3):
                # if we have overleaf results for too much time, we had better try it again
                R = np.array([[0,0]])
                R0 = np.array([[0,0]])
                r = np.zeros([1,2]) # [[x1,y1],[x2,y2].[x3,y3],...]    # where we store position data
                i = 0
                break
    ## return
    if target == "distance":
        x_end = r[-1,0] ; y_end = r[-1,1]
        return np.sqrt(x_end**2 + y_end**2)

    if target == "array":
        return r[:,0] , r[:,1] # xlist, ylist

def crw_operate_2(steps,a,target="array"): # constrained random walk with method 2
    ## operate
    Map = defaultdict(list) # Map which contains [x,y] and points belongs to it
    Map["[0,0]"].append(0) # initialize the Map

    PI = np.pi
    R = np.array([[0,0]])
    R0 = np.array([[0,0]])
    r = np.zeros([1,2]) # [[x1,y1],[x2,y2].[x3,y3],...]    # where we store position data
    
    i = 0 # step counter
    a2 = a**2 # turn a to a^2 

    while(i<steps):
        rec = 0 # remake counter
        while(1): 
            rec += 1

            p = 2*PI*np.random.rand()       # random angle
            R0 = R + np.array([[np.cos(p),np.sin(p)]])  # random walk

            x = int(np.round(R0[0,0])); y = int(np.round(R0[0,1])) # get X and Y of the R0          
            # get those points which we will judge in the next
            targetlist = []
            nx = x-2
            while (nx<=x+2):
                ny = y-2
                while (ny<=y+2):

                    catlist = Map[f"[{nx},{ny}]"]
                    for number in catlist:
                        targetlist.append(number)

                    ny += 1
                nx += 1
            
            Tar = np.array([targetlist])
            Mx = r[Tar,0];My = r[Tar,1]
            # if there is no overleaf we will receive this step and break to the next
            # else remake and rec += 1
            dist = (Mx.T - R0[0,0])**2 + (My.T - R0[0,1])**2
            dist = (dist < a2)
            f = np.sum(dist,axis = 0)

            # if there is no overleaf we will receive this step
            if f == 0:
                R = R0
                r = np.vstack((r,R))
                i += 1
                Map[f"[{x},{y}]"].append(i)
                break

            if rec > 1.5*(i+3):
                # if we have overleaf results for too much time, we had better try it again
                R = np.array([[0,0]])
                R0 = np.array([[0,0]])
                r = np.zeros([1,2]) # [[x1,y1],[x2,y2].[x3,y3],...]    # where we store position data
                i = 0
                # reset the Map
                del Map
                Map = defaultdict(list) # Map which contains [x,y] and points belongs to it
                Map["[0,0]"].append(0) # initialize the Map
                break
    ## return
    if target == "distance":
        x_end = r[-1,0] ; y_end = r[-1,1]
        return np.sqrt(x_end**2 + y_end**2)

    if target == "array":
        return r[:,0] , r[:,1] # xlist, ylist

if __name__ == "__main__":
    crw_main()
