import numpy as np
import matplotlib.pyplot as plt

def ao_main():
    c_const = 2.998*1e8
    n0 = 3.5
    a_i = [7.4969,1.9347]
    lbd_i = [0.4082,37.17]

    lbd_list = np.linspace(1.4,11,1000)
    n2 = n0*np.ones_like(lbd_list)
    dndl = np.zeros_like(lbd_list)
    dD = np.zeros_like(lbd_list)

    i = 0
    while(i<np.size(a_i)):
        n2 = n2 + a_i[i]*lbd_list**2/(lbd_list**2 - lbd_i[i]**2 )
        i += 1 
    n = np.sqrt(n2)

    i = 0
    while(i<np.size(a_i)):
        dndl = dndl - a_i[i]*lbd_i[i]**2/((lbd_list**2 - lbd_i[i]**2)**2)
        i += 1
    dndl = dndl*lbd_list/n

    N = n - lbd_list*dndl

    D = dndl/lbd_list - (dndl**2)/n
    i = 0
    while(i<np.size(a_i)):
        dD = dD + a_i[i]*lbd_i[i]**2/((lbd_list**2 - lbd_i[i]**2)**3)
        i += 1

    D = -(D + 2*lbd_list**2*dD/n)*(lbd_list/c_const)

    ## figure
    plt.subplot(1,2,1)
    plt.plot(lbd_list,n,label = "refractive index n",color = 'b')
    plt.plot(lbd_list,N,label = "group index N",color = 'r')
    plt.legend()
    plt.xlabel('wavelength/um')
    plt.title("refractive index & group index")

    plt.subplot(1,2,2)
    plt.plot(lbd_list,D)
    plt.xlabel('wavelength/um')
    plt.title("group velocity dispersion D_lambda")

    plt.show()

if __name__ == "__main__":
    ao_main()