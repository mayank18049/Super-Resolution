# !pip install scikit-image pystackreg

import numpy as np
import scipy as sc
from skimage import io, transform
from skimage.restoration import wiener
import matplotlib.pyplot as plt
from pystackreg import StackReg

#To implement a gaussian filter with geiven shape and std deviation-------------------
def gaussian_filter(shape, sigma):
    cm,cn = [(dim-1)//2 for dim in shape]
    rm = np.array(range(shape[0]))-cm
    rn = np.array(range(shape[1]))-cn
    a = 2*sigma**2
    x = np.exp(-(rm**2)/a).reshape(shape[0],1)
    y = np.exp(-(rn**2)/a).reshape(1,shape[1])
    filter = np.dot(x,y)/(np.pi*a)
    return filter

#Parameters for the model------------------------------------------------------------
n_img = 20		# Number of LR images
n_iter = 70		# Number of iterations for L1-Norm minimization
scale = 3		# Scale of HR Image (HR/LR)
beta = 0.6		# Learning Rate
lam = 0.05		# Blurring Kernel ratio 

#Reading images from the directory---------------------------------------------------
img_dir = 'dataset/'
ext = '.bmp'
imgs = []
for i in range(n_img):
    imgs.append(io.imread(img_dir+str(i+1)+ext))
imgs = np.array(imgs)


#Creating the image bluring filter with deconvolution by wiener method------------------------
# Kernel size calculation
kern = np.int(np.round(np.max(imgs[0].shape)*lam*scale))
blur = gaussian_filter((kern,kern), 1)
# Deconvolution using Wiener Filtering
dblimg = wiener(imgs[0],blur,balance=0)
# Point Spread Function estimation in fourier domain
blrft = np.fft.fftshift(np.fft.fft2(imgs[0]))/np.fft.fftshift(np.fft.fft2(dblimg))
blr = np.fft.ifft2(np.fft.ifftshift(np.abs(blrft)))
blr = np.fft.fftshift(np.abs(blr))
r = (imgs[0].shape[0]-1)//2
c = (imgs[0].shape[1]-1)//2
blur = blr[r-kern//2:r+kern//2+1,c-kern//2:c+kern//2+1]
blur /= blur.sum()
blur = gaussian_filter((kern,kern), 1)
# blur = np.ones((kern,kern))/kern**2
# print(blur/blur.sum())
io.imshow(np.uint8(blur*255))

# Image Regiseration using translation estimation and Affine Matrix Calculation
sr = StackReg(StackReg.TRANSLATION)
affmats = sr.register_stack(imgs,reference='first')

#Interpolation of Median average of LR Images as initial---------------------------------------
init = np.median(imgs, axis=0)
init = transform.rescale(init, scale, anti_aliasing=False)
io.imshow(np.uint8(init))

#Iterative L1-Norm Minimization for optimization of HR image-----------------------------------
X = np.copy(init)
invblur = np.flip(blur)
k = n_img
for i in range(n_iter):
    ML = np.zeros(X.shape)
    h,w = X.shape
    for j in range(k):
        tmpTmat = np.flip(affmats[j]).T
        tmpI = sr.transform(X,tmat=tmpTmat)
        # print(tmpI[0,0])
        ml = sc.signal.convolve2d(tmpI, blur, mode='same')
        # print(ml[0,0])
        x = np.array(range(0,h,scale)).reshape(imgs[j].shape[0],1)
        y = np.array(range(0,w,scale)).reshape(1,imgs[j].shape[1])

        xml = np.sign(ml[x,y] - imgs[j])
        xh,xw = np.int32(xml.shape)
        
        ml = np.zeros((h,w))
        for r in range(xh):
            for c in range(xw):
                ml[r*scale,c*scale] = xml[r,c]
        ml = sc.signal.convolve2d(ml, invblur, mode='same')
        ml = sr.transform(ml,tmat=affmats[j])
        # Cost = Dt*Ft*signum(DFX-Y) 
        ML += ml
    # Update step with gradients
    X -= beta*ML
    print("Iteration",i+1," of ",n_iter,": Cost: ",np.abs(ML.sum()/k))

io.imshow(np.uint8(init/init.max()*255))
plt.show()

io.imshow(np.uint8(X/X.max()*255))
plt.show()