import numpy as np
import numerical_integral as ni
import matplotlib.pyplot as plt 

## scalar diffraction theorem:
# g(x,y) = f(x,y) * h(x,y)
# h(x,y) = j/lambda*z e^{-jk*sqrt(x^2+y^2+z^2)}/sqrt(x^2+y^2+z^2)

def fun(ix,x,d,lbd):
    # g(x) = f(x)*h(x) = int_{-infty}^{+infty} f(x')*h(x-x') dx'
    # amplitude(x)*h(y-x)
    r = np.sqrt(d**2+(x-ix)**2)
    k = 2*np.pi/lbd
    f = np.exp(-(0+1j)*k*r)/(lbd*r**2)
    return f

def nd_main():
    # area [-infty,infty]
    # dimension = 1[um]
    lbd = 1.55 # 1.55um
    d = 2*1e3
    a = 2

    imageL = 400
    xlist = np.arange(-0.2*imageL,0.6*imageL+a,a)
    imagelist = np.empty_like(xlist,dtype = complex) ## important for complex
    i = 0
    while(i < np.size(xlist)):
        x = xlist[i]
        imagelist[i] = (0+1j)*d*ni.integral(fun,[0,np.infty],args = (x,d,lbd,),h_step = 0.1,TOL = 1e-10)
        i += 1

    imagelist = imagelist*1e12
    imagelist = np.abs(imagelist)**2
    imagelist = imagelist/(np.max(imagelist))

    ## figure
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(xlist,imagelist)
    plt.title('intensity plot')

    ln = np.size(imagelist)
    c = int(np.ceil(ln/1.414))
    image = np.zeros([c,ln])
    i = 0
    while (i<c):
        image[i,:] = imagelist
        i += 1
    plt.subplot(2,1,2)
    plt.imshow(image, cmap='gray', vmin=np.min(imagelist), vmax=np.max(imagelist),extent=[xlist[0], xlist[-1], 0, c])
    plt.tight_layout()
    plt.colorbar()  # 添加颜色条
    plt.title('blade diffraction pattern')
    plt.yticks([])
    plt.xticks([])
    plt.show()

if __name__ == '__main__':
    nd_main()
    #print((1j*1j))