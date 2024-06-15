import numpy as np
import matplotlib.pyplot as plt

def po_main():
    PI = np.pi
    ## users set
    w1 = 2*PI*4826 
    w2 = 2*PI*5582
    beta_1 = 2*PI*2.65
    beta_2 = 2*PI*13.82
    
    T = 80 # ms ## time of a modulation loop
    L = 76 # nm
    # f = 732.5 + 52.5*cos(2*PI*t/T) [Hz]
    # delta = 10 - 3.3*sin(2*PI*t/T) [nm]
    # L = L0 + delta*cos(2*PI*f*t)


if __name__ == "__main__":
    po_main()