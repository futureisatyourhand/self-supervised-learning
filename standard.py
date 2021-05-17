# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/5/12$ 12:12$
# @Author  : Qian Li
# @Email   : 1844857573@qq.com
# @File    : network.py
# Description : details(i.e., online network,online projector network, online predictor,classifier, target network, target projector,) for self-supervised learning
import torch
from functools import wraps
from torch import nn
import numpy as np
from utils import MLP,ResNet50
import copy
from torch.nn import init
from torchvision import models

def weigth_init(model,path):
    from collections import OrderedDict
    new_state_dict=OrderedDict()
    state_dict=torch.load(path)["model"]
    for k,v in state_dict.items():
        if "target_" in k:
            continue
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
 
    _, pred = output.topk(maxk, 1, True, True)  # 返回最大的k个结果（按最大到小排）
    pred = pred.t()  # 转置
    correct = pred.eq(target.view(1, -1).expand_as(pred))
 
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class VGG(nn.Module):
    def __init__(self,num_classes=10,
                        projector_hidden_size=4096,
                        projector_output_size=256,
                        predictor_hidden_size=4096,
                        moving_average_decay=.9999,
                        eps=1e-5,use_momentum = True,mode="pre-train"):
        ##model:pre-train,fine-tune,test
        super(VGG,self).__init__()       

        model=models.vgg16(pretrained=False)
        print(model)

        model.classifier=MLP(input_size=512,hidden_size=projector_hidden_size,output_size=projector_output_size)
        model.avgpool=nn.Sequential()
        self.mode=mode
        model.classifier=nn.Sequential()
        self.model=model
        self.classifier=nn.Sequential(nn.Linear(512,4096),
                                          nn.BatchNorm1d(4096),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(4096,4096),
                                          nn.BatchNorm1d(4096),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(4096,num_classes)
                                         )
        self.model=model
        self.cls_loss=nn.CrossEntropyLoss()

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
        if self.mode is "test":
            feature_view1=self.model(image_one)
            logits_view1=nn.Softmax(dim=1)(self.classifier(feature_view1))
            return logits_view1.argmax(dim=1),None,None
        feature=self.model(image_one)
        logit_view1=self.classifier(feature)        
        classifier_loss=self.cls_loss(logit_view1,labels)
        logit_view1=nn.Softmax(dim=1)(logit_view1)
        top1_acc,top5_acc=accuracy(logit_view1.data,labels, topk=(1, 5))
        return classifier_loss.mean(),top1_acc.data.mean(),top5_acc.data.mean()
