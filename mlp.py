# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/5/19$ 12:12$
# @Author  : Qian Li
# @Email   : 1844857573@qq.com
# @File    : mlp.py
# Description : mlp
import torch
from functools import wraps
from torch import nn
import numpy as np
from utils import MLP,ResNet50,External_attention,accuracy
import copy
from torch.nn import init
from torchvision import models

class VGGMlp(nn.Module):
    def __init__(self,num_classes=10,
                        eps=1e-5,use_momentum = True,mode="pre-train"):
        ##model:pre-train,fine-tune,test
        super(VGGMlp,self).__init__()       

        mdl=models.vgg16(pretrained=False).features
        #print(list(list(mdl.modules())[0].modules()))#lst=*list(model.modules())
        self.model=nn.Sequential(*list(mdl.modules())[1:10],*list(mdl.modules())[11:31])
        self.linu = External_attention(512)
        self.classifier=nn.Sequential(nn.Linear(8192,2048),
                                    nn.BatchNorm1d(2048),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(2048,512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512,num_classes),)
        
        self.cls_loss=nn.CrossEntropyLoss()
        self.mode=mode
        if self.classifier is not None:
            for m in self.classifier.modules():
                if isinstance(m,nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                elif isinstance(m,nn.Linear): 
                    init.normal_(m.weight, std=1e-3)
                elif isinstance(m,nn.BatchNorm2d):
                    init.constant(m.weight, 1)
                    init.constant(m.bias, 0)  
                elif isinstance(m,nn.BatchNorm1d):
                    init.constant(m.weight, 1)
                    init.constant(m.bias, 0)

                
    def forward(self,image_one=None,image_two=None,labels=None):
        #if not image_two:
        feature=self.model(image_one)
        feature=self.linu(feature)
        feature=feature.view(image_one.shape[0],-1)
        feature=self.classifier(feature) 
        if self.mode is "test":
            return nn.Softmax(dim=1)(feature).argmax(dim=1),None,None
        classifier_loss=self.cls_loss(feature,labels)
        feature=nn.Softmax(dim=1)(feature)
        top1_acc,top5_acc=accuracy(feature.data,labels, topk=(1, 5))
        return classifier_loss.mean(),top1_acc.data.mean(),top5_acc.data.mean()
