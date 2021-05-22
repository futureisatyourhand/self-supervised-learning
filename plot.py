import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
files=open("vgg_log64.txt",'r')
files_byol=open("byol_log64.txt",'r')
mlp_files=open("mlp_log64.txt",'r')
byol_mlp_files=open("byol_mlp_log64.txt",'r')
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


##+++++++++++++++++++++++++++++
mlp=[]
i=0
log=0.0
for f in mlp_files.readlines()[1:]:
    f=f.strip('\n').split(',')
    if len(f)>4:
        continue
    if i==100:
        log/=100.
        mlp.append(log)
        log=float(f[1])
        i=0
    else:
        i+=1
        log+=float(f[1])
mlp_files.close()
mlp=np.array(mlp)



##+++++++++++++++++++++++++++++
byol_mlp=[]
i=0
log=0.0
for f in byol_mlp_files.readlines()[1:]:
    f=f.strip('\n').split(',')
    if len(f)>4:
        continue
    if i==100:
        log/=100.
        byol_mlp.append(log)
        log=float(f[1])
        i=0
    else:
        i+=1
        log+=float(f[1])
byol_mlp_files.close()
byol_mlp=np.array(byol_mlp)
print(byol.shape,vgg.shape)

#x_smooth = np.linspace(vgg.min(), vgg.max(), 300)
#y_smooth = make_interp_spline(vgg, x)(x_smooth)
plt.plot(x, vgg,color='#0000FF',label='standard vgg')
plt.plot(x, byol,color='#FF000F',label='byol')
plt.plot(x, mlp,color='#00FF00',label='MLP for standard vgg')
plt.plot(x, byol_mlp,color='#000F0F',label='byol with mlp')
plt.xlabel('iterations')
plt.ylabel('per-iter loss')
plt.legend()
plt.show()