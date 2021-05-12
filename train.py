# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/5/12$ 12:12$
# @Author  : Qian Li
# @Email   : 1844857573@qq.com
# @File    : train.py
# Description :train and eval for self-supervised learning,such as CIFAR10,CIFAR100,STL and etc.
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
parser.add_argument('--batch', type=int, required = True,default=64,
                       help='batch size for self-supervised learning')

###########
#init
torch.distributed.init_process_group(backend="nccl")
##distribute process
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

transform = transforms.Compose([
                transforms.RandomResizedCrop((224,224), 
                        scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), 
                        interpolation=im_.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
               ])
im_train_list=glob.glob("/home/liqian/data/cifar10/cifar-imgs/train/*/*.png")
im_test_list=glob.glob("/home/liqian/data/cifar10/cifar-imgs/test/*/*.png")
trainset = Data(im_train_list,train = True,trans=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=False,num_workers=4, sampler=DistributedSampler(trainset))

testset = Data(im_test_list,train=False,trans=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=32,shuffle=True,num_workers=4)
test_len=len(testloader)
classes = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')


###########
#init
#torch.distributed.init_process_group(backend="nccl")
##distribute process
#local_rank = torch.distributed.get_rank()
#torch.cuda.set_device(local_rank)
#device = torch.device("cuda", local_rank)

model=BYOL(mode="train+val")
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

for epoch in range(300):
    #train
    iter_loss=0.0
    iter_top1=0.0
    iter_top5=0.0
    
    for i,data in enumerate(trainloader,0):
        inputs1,inputs2,labels=data
        inputs1,inputs2, labels = Variable(inputs1.cuda()),Variable(inputs2.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        loss,top1,top5=model(inputs1,inputs2,labels)
        loss.backward()
        optimizer.step()
        iter_loss+=loss.item()
        iter_top1+=top1
        iter_top5+=top5
        if i%10==0:
            print('[%d epoch, %5d iter] || loss: %.3f | top1:%.3f | top5:%.3f'%(epoch+1,i+1,iter_loss/10.,iter_top1/10.,iter_top5/10.))
            iter_loss=0.0
            iter_top1=0.0
            iter_top5=0.0

    #validate
    if epoch%20==0:
        iter_loss=0.0
        iter_top1=0.0
        iter_top5=0.0
        for i,data in enumerate(testloader,0):
            inputs,labels=data
            with torch.no_grad():
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                _,top1,top5=model(inputs,None,labels)
                #iter_loss+=loss.item()
                iter_top1+=top1
                iter_top5+=top5
        print('top1:%.3f | top5:%.3f'%(iter_top1/test_len,iter_top5/test_len))

#testing    
iter_loss=0.0
iter_top1=0.0
iter_top5=0.0
for i,data in enumerate(testloader,0):
    inputs,labels=data
    with torch.no_grad():
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        _,top1,top5=model(inputs,None,labels)
        #iter_loss+=loss.item()
        iter_top1+=top1
        iter_top5+=top5
print('top1:%.3f | top5:%.3f'%(iter_top1/test_len,iter_top5/test_len))
               
