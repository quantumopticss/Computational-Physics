import numpy as np
import matplotlib.pyplot as plt

def ef_T(mesh,source):
    1

def ef_main():
    ## users set
    h = 0.01  # discret size
    r = 5;c=5 # total size

    q1 = 1; q2 = -1


    nx = int(r/h);ny = int(c/h)
    mesh = np.zeros(nx+2,ny+2)
    source = np.zeros_like(mesh)

    ## figrue
    plt.figure(1)
    plt.title("E_field")


if __name__ == "__main__":
    ef_main()
