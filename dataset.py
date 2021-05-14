# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/5/12$ 12:12$
# @Author  : Qian Li
# @Email   : 1844857573@qq.com
# @File    : dataset.py
# Description :dataset and multi-views augmentations for self-supervised learning
from PIL import Image as im_
from PIL import ImageOps as imo
from PIL import ImageEnhance as ime
from PIL import ImageFilter as imf
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import random
label_name = ["airplane", "automobile", "bird",
              "cat", "deer", "dog",
              "frog", "horse", "ship", "truck"]

label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx

class Data(Dataset):
    def __init__(self,im_list,trans=None,train=True,epochs=300):
        super(Dataset,self).__init__()
        imgs=[]
        for im_item in im_list:
            im_label_name=im_item.split('/')[-2]
            imgs.append([label_dict[im_label_name],im_item])
        #self.imgs=imgs
        images=[]
        for i in range(epochs):
            random.shuffle(imgs)
            images.extend(imgs)
        del imgs
        self.imgs=images
        self.trans=trans
        self.tran=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225)),])
        self.train=train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        label,image=self.imgs[idx]
        image=Image.open(image)
        
        if self.train:
            view1=BYOLAugmentationsView1(image)
            view2=BYOLAugmentationsView2(image)
            return self.tran(view1),self.tran(view2),label
        else:
            return self.trans(image),label

def BYOLAugmentationsView1(img):
    
    t1 = transforms.Compose([transforms.RandomResizedCrop((32,32), scale=(
        0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=im_.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5)])
    img = t1(img)

    if np.random.uniform() < 0.8:
        t2 = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        img = t2(img)

    if np.random.uniform() < 0.2:
        img = transforms.functional.to_grayscale(img, num_output_channels=3)

    img = img.filter(imf.GaussianBlur(
            radius=np.random.uniform(0.1, 2.0)))
    #if np.random.uniform() < 0.2:
    #    t3 = transforms.Lambda(lambda x: imo.solarize(x,0.5))
    #    img = t3(img)
    #    print(np.array(img))

    return img


def BYOLAugmentationsView2(img):

    t1 = transforms.Compose([transforms.RandomResizedCrop((32,32), scale=(
        0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=im_.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),])
    img = t1(img)
    
    #print(np.array(img).shape)
    #exit()
    if np.random.uniform() < 0.8:
        t2 = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        img = t2(img)

    if np.random.uniform() < 0.2:
        img = transforms.functional.to_grayscale(img, num_output_channels=3)
    if np.random.uniform()<0.1:
        img = img.filter(imf.GaussianBlur(
            radius=np.random.uniform(0.1, 2.0)))

    #if np.random.uniform() < 0.2:
    #    t3 = transforms.Lambda(lambda x: imo.solarize(x,0.5))
    #    img = t3(img)
     

    return img

