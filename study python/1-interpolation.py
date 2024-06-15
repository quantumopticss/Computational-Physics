import numpy as np
import matplotlib.pyplot as plt

def interpolation_operator(X,i,nx,xlist,ylist):
    result = np.ones_like(X)

    j = 0
    while(j<nx):
        if j == i:
            j += 1;continue
        result *= ( (X-xlist[j])/(xlist[j] - xlist[i]) )
        j += 1 
    
    result *= ylist[i]

    return result


def interpolation_main():
    
    xlist = np.array([-2,-0.5,1,1.2,2])
    ylist = np.array([-1,0,1,1.2,1.8]) 

    X = np.linspace(-3,3,1000,endpoint = True)
    nx = np.size(xlist);ny = np.size(ylist)

    if nx != ny:
        print('error,nx!=ny');return
    
    IX = np.zeros_like(X)
    i = 0
    while(i<nx):
        IX += interpolation_operator(X,i,nx,xlist,ylist)
        i += 1

    ## figure
    plt.figure(1)
    plt.plot(X,IX,label = 'interpolation')
    plt.scatter(xlist,ylist,label = 'data point')
    plt.title('interpolation')
    plt.show()




if __name__ == "__main__":
    interpolation_main()
