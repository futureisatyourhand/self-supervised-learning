# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/5/12$ 12:12$
# @Author  : Qian Li
# @Email   : 1844857573@qq.com
# @File    : plot.py
# Description :plot trainingfor self-supervised learning or standard training,such as CIFAR10,CIFAR100,STL and etc.
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
files=open("vgg_log64.txt",'r')
files_byol=open("byol_log64.txt",'r')
vgg=[]
i=0
log=0.0
for f in files.readlines()[1:]:
    f=f.strip('\n').split(',')
    if len(f)>4:
        continue
    if i==100:
        log/=100.
        vgg.append(log)
        log=float(f[1])
        i=0
    else:
        i+=1
        log+=float(f[1])
vgg=np.array(vgg)
print(vgg)
x=np.arange(vgg.shape[0])
######=====================
byol=[]
i=0
log=0.0
for f in files_byol.readlines()[1:]:
    f=f.strip('\n').split(',')
    if len(f)>4:
        continue
    if i==100:
        log/=100.
        byol.append(log)
        log=float(f[1])
        i=0
    else:
        i+=1
        log+=float(f[1])
files_byol.close()
byol=np.array(byol)
print(byol.shape,vgg.shape)

#x_smooth = np.linspace(vgg.min(), vgg.max(), 300)
#y_smooth = make_interp_spline(vgg, x)(x_smooth)
plt.plot(x, vgg,color='#0000FF',label='standard vgg')
plt.plot(x, byol,color='#FF0000',label='byol')
plt.xlabel('iterations')
plt.ylabel('per-iter loss')
plt.legend()
plt.show()
