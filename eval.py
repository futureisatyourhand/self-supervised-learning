# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/5/12$ 12:12$
# @Author  : Qian Li
# @Email   : 1844857573@qq.com
# @File    : eval.py
# Description :eval model forself-supervised learning and standard training,such as CIFAR10,CIFAR100,STL and etc.
import random
import glob
from network import BYOL
from dataset import Data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torchvision
import numpy as np
from PIL import Image as im_
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
from torch import optim
import argparse
from dataset import Data
from network import weigth_init
from standard import VGG
##argparser
parser = argparse.ArgumentParser(description='my byol')
parser.add_argument('--batch', type=int, required = True,default=32,
                       help='batch size for self-supervised learning')
#args = parser.parse_args()
#####seed###
random.seed(120)
np.random.seed(122) # for yolov5-mosaic
torch.manual_seed(120)  
torch.cuda.manual_seed(120) 
torch.cuda.manual_seed_all(120)
#####
batch=64

###########
#init
torch.distributed.init_process_group(backend="nccl")
##distribute process
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225)),
               ])
im_test_list=glob.glob("/home/liqian/data/cifar10/cifar-imgs/test/*/*.png")
testset = Data(im_test_list,train=False,trans=transform,epochs=1)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch,shuffle=True,num_workers=4,method="vgg")
test_len=len(testloader)
classes = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
####################+++++++++++++++++++++++++++++++
method=2
if method==1:
    model=BYOL(mode="test")
else:
    model=VGG(mode="test")
weigth_init(model,"models/64/model_classifier_117000.pth")
model.to(device)
model.eval()
###optimizer
#model=torch.nn.parallel.DistributedDataParallel(model)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
#testing    
acc=0.0
for i,data in enumerate(testloader,0):
    inputs,labels=data
    with torch.no_grad():
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        result,_,_=model(inputs,None,labels)
        acc+=(result==labels).float().mean()
        
print('accuracy:%.3f'%(acc/test_len))

#torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':300},"models/"+str(batch)+"/model_final.pth")
               
