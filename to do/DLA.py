import numpy as np
import matplotlib.pyplot as plt

def dla_statistic(A,n):
    rlist = np.arange(1,(n//2),3)
    dlist = np.empty_like(rlist)
    f = rlist.size
    m = 0

    while (m<f):
        Mat = A[n//2-rlist[m]:n//2+rlist[m]+1,n//2-rlist[m]:n//2+rlist[m]+1]
        dlist[m] = np.sum(Mat)

        m += 1

    return rlist,dlist

def dla_run(A,n,eta = 1):
    ## start
    s = np.random.randint(1,4)
    if s == 1:
        r = 1; c = np.random.randint(1,n-2) 
    elif s == 2:
        r = n-2;c = np.random.randint(1,n-2)
    elif s == 3:
        r = np.random.randint(1,n-2); c = 1
    else:
        r = np.random.randint(1,n-2); c = n-2

    ## run
    while (1):
        if (A[r+1,c] or A[r-1,c] or A[r,c+1] or A[r,c-1]) and (np.random.random() <= eta):
            A[r,c] = 1
            return A
        p = np.random.random()
        if p <= 0.25:
            r += 1
        elif p <= 0.5:
            r -= 1
        elif p <= 0.75:
            c += 1
        else:
            c -= 1

        # restart
        if r == 0 or r == n-1 or c == 0 or c==n-1:
            s = np.random.random()
            if s <= 0.25:
                r = 1; c = np.random.randint(1,n-2) 
            elif s <= 0.5:
                r = n-2;c = np.random.randint(1,n-2)
            elif s <= 0.75:
                r = np.random.randint(1,n-2); c = 1
            else:
                r = np.random.randint(1,n-2); c = n-2
            
def dla_calculate(n,Num,eta = 1):
    ## users set 
    # n -- simulation dimension 
    # Num -- number of total particle input
    # eta -- DLA stickness parameter

    ## start
    A = np.zeros([n,n],dtype = int)
    A[n//2,n//2] = 1

    ## run
    while (Num):    
        # run dla_
        dla_run(A,n,eta)
        Num -= 1

    A = A > 0 # reshape
    return A

def dla_main():
    ## users set
    n = 155 # -- simulation dimension 
    Num = 1500 #-- number of total particle input
    eta  = 1 #-- DLA stickness parameter

    A = dla_calculate(n,Num,eta)

    ## statistic
    rlist, dlist = dla_statistic(A,n)

    ## figure
    plt.figure(1)
    plt.imshow(A)
    plt.title("DLA figure")
    plt.show()

    plt.figure(2)
    rr = np.linspace(0,np.log(n/2-1),500)
    plt.plot(np.log(rlist),np.log(dlist),label = "Dimension")
    plt.plot(rr,1+rr*1.65,label = "curv fitting")
    plt.legend()
    plt.xlabel("radius of the square")
    plt.show()

if __name__ == "__main__":
    dla_main()
