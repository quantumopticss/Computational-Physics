import numpy as np
import matplotlib.pyplot as plt
import DLA

def sDLA_d_main():
    ## users set
    n = 180 # dimension of simulation area
    Num = 1700 # particles to input

    ## eta list 
    eta_list = np.linspace(0.1 ,1,6,endpoint = True)
    d_result = np.empty_like(eta_list)

    ## run
    i = 0; L = np.size(eta_list)
    while (i<L):
        A = DLA.dla_calculate(n,Num,eta_list[i])
        rlist,dlist = DLA.dla_statistic(A,n)

        # fit and result
        k,b = np.polyfit(np.log(rlist),np.log(dlist),1) # y = kx + b 
        d_result[i] = k

        i += 1

    ## figrue
    plt.figure(1)
    plt.plot(eta_list,d_result)
    plt.title("Relation between stickness and dimension")
    plt.xlabel("Stickness")
    plt.show()

if __name__ == "__main__":
    sDLA_d_main()