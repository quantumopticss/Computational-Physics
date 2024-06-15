import numpy as np
import matplotlib.pyplot as plt

def rw_run(steps):
    location = np.zeros([1,2])
    data_list = np.empty([steps,2],dtype = float)
    stp = 0
    # random walk operation
    while (stp<steps):

        data_list[stp,:] = location
        stp += 1

        # random walk
        s = np.random.rand()
        if s<0.25: 
            location[0,0] += 1
        elif s<0.5:
            location[0,0] += (-1)
        elif s <0.75:
            location[0,1] += 1
        else:
            location[0,1] += (-1)     
    
    return data_list

def rw_statistic(data_list):
    p_data = np.power(data_list[:,0],2) + np.power(data_list[:,1],2) # distance^2
    return p_data

def rw_main():
    # users set
    num = 200; n = 0 # average of N
    steps = 1000 # total steps

    result = np.empty([num,steps])

    # run times num
    while(n<num):
        data_list = rw_run(steps)
        result[n,:] = rw_statistic(data_list)

        n += 1

    # cal
    RMS_d = np.power((np.mean(result,0)),1/2)

    # figure
    x_axis = np.arange(steps)
    plt.figure(1)
    plt.plot(x_axis,RMS_d,label = "Simulation");plt.plot(x_axis,np.power(x_axis,1/2),label = "1/2 power")
    plt.title("Random Walk")
    plt.ylabel("RMS Distance");plt.xlabel("Steps")
    plt.legend()
    plt.show()

rw_main()
