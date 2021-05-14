# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/5/12$ 12:12$
# @Author  : Qian Li
# @Email   : 1844857573@qq.com
# @File    : train.py
# Description :pre-train for self-supervised learning (BYOL),such as CIFAR10,CIFAR100,STL and etc.
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
from dataset import Data,BYOLAugmentationsView1,BYOLAugmentationsView2
from network import weigth_init
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
                #transforms.RandomResizedCrop((224,224), 
                #        scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), 
                #        interpolation=im_.BICUBIC),
                #transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225)),
                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
               ])
im_train_list=glob.glob("/home/liqian/data/cifar10/cifar-imgs/train/*/*.png")
im_test_list=glob.glob("/home/liqian/data/cifar10/cifar-imgs/test/*/*.png")
trainset = Data(im_train_list,train = True,trans=transform,epochs=300)
sampler=DistributedSampler(trainset,shuffle=True)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch,shuffle=False,num_workers=4, sampler=sampler)

testset = Data(im_test_list,train=False,trans=transform,epochs=1)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch,shuffle=True,num_workers=4)
test_len=len(testloader)
classes = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')


###########
#init
#torch.distributed.init_process_group(backend="nccl")
##distribute process
#local_rank = torch.distributed.get_rank()
#torch.cuda.set_device(local_rank)
#device = torch.device("cuda", local_rank)

model=BYOL(mode="pre-train")
#weigth_init(model)
model.to(device)

###optimizer
optimizer=optim.SGD(model.parameters(),lr = 0.03,momentum=0.9,weight_decay=0.0001)

#model=torch.nn.parallel.DistributedDataParallel(model)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
####batch size:64 iter:780

logs=open("logs/pretrain_log"+str(batch)+".txt",'a+')
for epoch in range(1):
    #train
    iter_loss=0.0
    sampler.set_epoch(epoch)    
    for i,data in enumerate(trainloader,0):
        inputs1,inputs2,labels=data
        inputs1,inputs2, labels = Variable(inputs1.cuda()),Variable(inputs2.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        loss,top1,top5=model(inputs1,inputs2,labels)
        loss.backward()
        optimizer.step()
        model.update_target()
        iter_loss+=loss.item()
        if i%10==0:
            print('[%5d iter] || loss: %.3f '%(i+1,iter_loss/10.))
            logs.write(str(i+1)+","+str(iter_loss/10.)+"\n")
            iter_loss=0.0
            if i%39000==0:
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':i%39000},"models/"+str(batch)+"/model_"+str(i)+".pth")
logs.close()
#testing    

torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':300},"models/"+str(batch)+"/model_final.pth")
               
