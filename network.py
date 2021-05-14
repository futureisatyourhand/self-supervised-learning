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
#byol's loss function:L2-normalization
def loss_fn(x,y):
    x=nn.functional.normalize(x,dim=-1,p=2)
    y=nn.functional.normalize(y,dim=-1,p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new 
        return old * self.beta + (1 - self.beta) * new 

def weigth_init(model):
    for m in model.modules():
        print(m)
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data,0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0,0.01)
            m.bias.data.zero_()

def update_moving_average(ema_updater, ma_model1, current_model1):
    for current_params, ma_params in zip(current_model1.parameters(), ma_model1.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)
def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

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

class BYOL(nn.Module):
    def __init__(self,num_classes=10,
                        projector_hidden_size=4096,
                        projector_output_size=256,
                        predictor_hidden_size=4096,
                        moving_average_decay=.9999,
                        eps=1e-5,use_momentum = True,mode="pre-train"):
        ##model:pre-train,fine-tune,test
        super(BYOL,self).__init__()       

        model=models.vgg16(pretrained=False)
        print(model)

        model.classifier=MLP(input_size=25088,hidden_size=projector_hidden_size,output_size=projector_output_size)
        model.classifier=nn.Sequential()
        self.online_backbone=model#nn.Sequential(*list(model.modules())[:-1])
        #self.online_backbone.fc=nn.Sequential()
        #for param in self.online_backbone.parameters():
        #    param.requires_grad = False
        self.online_projector=MLP(input_size=25088,hidden_size=projector_hidden_size,output_size=projector_output_size)
        self.online_predictor=MLP(input_size=projector_output_size,hidden_size=predictor_hidden_size,output_size=projector_output_size)

        self.target_backbone = None
        self.target_projector=None
        self.mode=mode
        self.classifier=None
        if mode not in "pre-train":
            self.classifier=nn.Linear(25088,num_classes,bias=True)
            self.cls_loss=nn.CrossEntropyLoss()

        self.ema=EMA(moving_average_decay)
        self.use_momentum = use_momentum
        
        for m in self.online_projector.modules():
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
   
        for m in self.online_predictor.modules():
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

    def update_target(self):
        assert self.target_backbone is not None,'target backbone has not been created yet'
        assert self.target_projector is not None,"target projector has not been created yet!"
        update_moving_average(self.ema,self.target_backbone,self.online_backbone)
        update_moving_average(self.ema,self.target_projector,self.online_projector)

    @singleton('target_backbone')
    def _get_target_backbone(self):
        target_backbone = copy.deepcopy(self.online_backbone)
        for p in target_backbone.parameters():
            p.requires_grad=False
        return target_backbone 
    @singleton('target_projector')
    def _get_target_projector(self):
        target_projector=copy.deepcopy(self.online_projector)
        for p in target_projector.parameters():
            p.requires_grad=False
        return target_projector

    def freeze(self,checkpoint):
        if self.mode is not "pre-train":            
            ####train self-supervised(i.e.,BYOL)
            for p in self.online_backbone.parameters():
                p.requires_grad=False
            for p in self.online_predictor.parameters():
                p.requires_grad=False
            for p in self.online_projector.parameters():
                p.requires_grad=False
            if not self.target_backbone:
                for p in self.target_backbone.parameters():
                    p.requires_grad=False
            if not self.target_projector:
                for p in self.target_projector.parameters():
                    p.requires_grad=False
        else:
            for p in self.classifier.parameters():
                p.requires_grad=False
                
    def forward(self,image_one=None,image_two=None,labels=None):
        #if not image_two: 
            
        ##get view1
        feature_view1=self.online_backbone(image_one)
        projector_view1=self.online_projector(feature_view1)
        predictor_view1=self.online_predictor(projector_view1)
        if self.mode not in "pre-train" and image_two is None:
            logits_view1=nn.Softmax(dim=1)(self.classifier(feature_view1))
            top1_acc,top5_acc=accuracy(logits_view1.data,labels, topk=(1, 5))
            return None,top1_acc.data.mean(),top5_acc.data.mean() 
        ## get view2
        feature_view2=self.online_backbone(image_two)
        projector_view2=self.online_projector(feature_view2)
        #feature_view2=feature_view2.view(image_two.size(0),-1)
        predictor_view2=self.online_predictor(projector_view2)

        with torch.no_grad():
            if self.use_momentum:
                target_backbone = self._get_target_backbone()
                target_projector=self._get_target_projector()
            else:
                target_backbone=self.online_backbone
                target_projector=self.online_projector

            target_f1=target_backbone(image_one)
            target_f1=target_projector(target_f1)
            target_f2=target_backbone(image_two)
            target_f2=target_projector(target_f2)
            target_f1.detach_()
            target_f2.detach_()

        loss_one=loss_fn(predictor_view1,target_f2.detach()).mean()
        loss_two=loss_fn(predictor_view2,target_f1.detach()).mean()
        loss=loss_one+loss_two
        #self.update_target()
        if self.mode not in "pre-train":
            #feature_view1.detach_()
            #print(feature_view1.shape)
            logit_view1=self.classifier(feature_view1)
            classifier_loss=self.cls_loss(logit_view1,labels)
            loss+=classifier_loss.mean()
            logit_view1=nn.Softmax(dim=1)(logit_view1)#.cpu().detach().numpy().argsort(axis=1)[:,::-1]
            top1_acc,top5_acc=accuracy(logit_view1.data,labels, topk=(1, 5))
            #print(logit_view1)
            #labels=labels.data.cpu().detach().numpy()
            #top1_acc=np.array(logit_view1[:,:1]==np.tile(labels[:,np.newaxis],[1,1]),dtype=np.int).mean()
            #top5_acc=np.array(logit_view1[:,:5]==np.tile(labels[:,np.newaxis],[1,5]),dtype=np.int).mean()
            return loss,top1_acc.data.mean(),top5_acc.data.mean()
        else:
            #loss=loss_one+loss_two
            return loss,None,None
