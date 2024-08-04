## *args 是打包和解包
# 在传递args的时候不加*
# 例如integral接受调用处的 args = (a,b,...)
# 传递给 integral_calculate(...,args,...)
# integral_calculate 调用 func 需要解包，fun(...,*args,...)
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
fun = lambda x: np.cos(2*5*pi*x) + np.cos(2*pi*5.2*x)
tspan = 5; s_freq = 20

tlist = np.arange(0,tspan,1/s_freq)
ylist = fun(tlist)

fy = np.fft.fftshift(np.fft.fft(ylist))
freq_y = np.arange(-len(ylist)//2,len(ylist)//2)/tspan 

yylist = np.hstack((ylist,np.array([0]*len(ylist)*3)))
fyy = np.fft.fftshift(np.fft.fft(yylist))
freq_yy = np.arange(-len(yylist)//2,len(yylist)//2)/(2*tspan)

plt.subplot(1,2,1)
plt.plot(freq_y,np.abs(fy))
plt.title('raw')

plt.subplot(1,2,2)
plt.plot(freq_yy,np.abs(fyy))
plt.title('double 0')

plt.show()
