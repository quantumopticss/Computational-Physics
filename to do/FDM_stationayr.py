import numpy as np
import matplotlib.pyplot as plt

def fdm_start(nr,nc,boundary):
    A = np.zeros([nr,nc])

    A[0:2,:] = boundary[0]
    A[-2:,:] = boundary[1]
    A[:,0:2] = boundary[2]
    A[:,-2:] = boundary[3]

    return A

def fdm_calculate(A,nr,nc,accuracy=1): 
    # core = A[2:nr+2,2:nc+2]
    core_d2 = A[4:,2:nc+2] 
    core_u2 = A[0:nr,2:nc+2]
    core_l2 = A[2:nr+2,0:nc]
    core_r2 = A[2:nr+2,4:]

    core_d1 = A[3:nr+3,2:nc+2]
    core_u1 = A[1:nr+1,2:nc+2]
    core_l1 = A[2:nr+2,1:nc+1]
    core_r1 = A[2:nr+2,3:nc+3]

    if accuracy == 1:
        core = (core_l1+core_r1+core_u1+core_d1)/4 # first order accuracy
    else:
        core = (core_l1+core_r1+core_u1+core_d1)/4 # first order accuracy
        core2 = (16*core_l1 + 16*core_r1 - core_l2 - core_r2 + 16*core_d1 + 16*core_u1 - core_d2 - core_u2)/60 # second order accuracy
        core[2:nr-2,2:nc-2] = core2[2:nr-2,2:nc-2] 

    A[2:nr+2,2:nc+2] = core
    return A


def fdm_operate(hight,width,h,boundary,accuracy=1):

    ## calculate ***************************************************
    # initialize
    nr = int(hight/h)
    nc = int(width/h)
    A = fdm_start(nr+4,nc+4,boundary) # 2-n+1  (0,1) ----- (n+2,n+3)

    # calculate_average
    i = 2
    while(i<(nr+2)):
        j = 2
        while(j<(nc+2)):
            A[i,j] = ( (A[i,0]*(nc+2-j) + A[i,-1]*(j-1))/(nc+1) + (A[0,j]*(nr+2-i) + A[-1,j]*(i-1))/(nr+1) )/2
            j += 1
        i += 1
    # calculate_laplaciation
    num = 0.5*np.sqrt(i*j); del i,j
    while(num>0):
        A = fdm_calculate(A,nr,nc,accuracy)
        num -= 1
    
    return A

def fdm_main():
    ## users set**************************************************
    # geometry
    hight = 1;width = 1 # size of the area
    h = 0.002 # size of the mesh

    # accuracy
    accuracy = 1 # 1 or 2

    # boundary condition
    b_up=0;b_down=40;b_left=0;b_right = 40
    boundary = np.array([b_up,b_down,b_left,b_right])

    ## result
    A = fdm_operate(hight,width,h,boundary,accuracy) 

    ## figure ******************************************************
    nr = int(hight/h)
    nc = int(width/h)
    A = A[2:nr+2,2:nc+2] # 不包括停止索引
    plt.figure(1)
    plt.imshow(A)
    plt.title("Laplaciation_stationary")
    plt.xlabel("width");plt.ylabel("hight")
    plt.colorbar()
    plt.show();plt.close()


if __name__ == "__main__":
    fdm_main()