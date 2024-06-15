import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,50,endpoint = True)

plt.subplot(1,2,1)
plt.loglog(x,x**2,"g.",label = "quadratic")
plt.xlabel("X axis",fontsize = 18)
plt.legend(loc = 2) # legend label
plt.title("loglog plot")
plt.xlim(-6,6) # range of x axis
plt.grid() # mesh grid

plt.subplot(1,2,2)
plt.plot(x,x**3,'b-',label = "cubic")
plt.xlabel("X axis",fontsize = 18)
plt.legend(loc = 2) # legend label
plt.title(f"plot polynomials functions from {x[0]} to {x[-1]}")
plt.xlim(-6,6) # range of x axis
plt.grid() # mesh grid

plt.savefig("image.pdf")

# plt.loglog
# plt.semilogx
# plt.semilogy