import numpy as np
import matplotlib.pyplot as plt

def rw_run(Ndim,steps):
    location = np.zeros([Ndim])
    data_list = np.empty([Ndim,steps],dtype = float)
    stp = 0
    # random walk operation
    while (stp<steps):

        data_list[:,stp] = location
        stp += 1

        # random walk
        s = np.random.randint(0,Ndim-1)
        p = 1 if np.random.rand() > 0.5 else -1

        location[s] = location[s] + p
    
    return data_list

def rw_statistic(data_list):
    p_data = np.sum(data_list**2,axis = 0)
    return p_data

def rw_main():
    # users set
    Ndim = 3
    num = 200; n = 0 # average of N
    steps = 2000 # total steps

    result = np.empty([num,steps])

    # run times num
    while(n<num):
        data_list = rw_run(Ndim,steps)
        result[n,:] = rw_statistic(data_list)

        n += 1

    # cal
    RMS_d = np.power((np.mean(result,0)),1/2)

    # figure
    x_axis = np.arange(steps)
    plt.figure(1)
    plt.plot(x_axis,3.2*RMS_d,label = "Simulation")
    plt.plot(x_axis,x_axis**((Ndim-1)/Ndim),label = "Simulation")
    plt.title("Random Walk")
    plt.ylabel("RMS Distance");plt.xlabel("Steps")
    plt.legend()
    plt.show()

rw_main()
