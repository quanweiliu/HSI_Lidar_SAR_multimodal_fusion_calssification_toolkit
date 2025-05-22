# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:52:59 2025

@author: Leo
"""
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchvision.models.resnet import resnet18
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights
# from torchvision.models.resnet import resnet34
# from torchvision.models.resnet import resnet50
from torchvision.models import wide_resnet50_2

from torchvision.models import resnext50_32x4d
from torchvision.models import densenet161
#from cbam_resnext import cbam_resnext50_16x64d


# 来源于 ACNet, 我自己魔改的 ResNet，需要记住的是，
# resNet 在大多数数据集上没有 vit 好，特别是数据量比较少的时候
nonlinearity = partial(F.relu, inplace=True)

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    


class Model_base(nn.Module):
    def __init__(self, input_dim, backbone='resnet18', is_pretrained=False):
        super(Model_base, self).__init__()
        if backbone == 'resnet18':
            filters = [64, 128, 256, 512]  # ResNet18
            resnet = resnet18(weights=None if not is_pretrained else 'DEFAULT')
        elif backbone == 'resnet34':
            filters = [64, 128, 256, 512]  # ResNet34
            resnet = resnet34(weights=None if not is_pretrained else 'DEFAULT')
        elif backbone == 'resnet50':
            filters = [256, 512, 1024, 2048]  # ResNet50
            resnet = resnet50(weights=None if not is_pretrained else 'DEFAULT')
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        # print(resnet)

        # rgb-decoder
        self.first = nn.Sequential(
            ConvBNReLU(input_dim, filters[0], ks=3, stride=1, padding=1),
        )

        self.encoder1 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            resnet.layer1
        )
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.avgpool = resnet.avgpool

    def forward(self, x):
        x_first = self.first(x)
        xe1 = self.encoder1(x_first)
        xe2 = self.encoder2(xe1)
        xe3 = self.encoder3(xe2)
        xe4 = self.encoder4(xe3)

        out = self.avgpool(xe4)
        out = torch.flatten(out, start_dim=1)

        return out


# # Chatgpt 给我写的一个可以复用权重的版本。，我发现不需要删除 maxpooling 层，
# # 整个resNet 不是靠 maxpool 减小尺寸的，只有最开始的时候有个 maxpooling.
# class Model_base(nn.Module):
#     def __init__(self, input_dim):
#         super(Model_base, self).__init__()
        
#         # 加载预训练 ResNet18
#         resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
#         # 替换第一层 Conv2d
#         resnet.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=True)
        
#         # # 递归替换 MaxPool2d 层
#         # def remove_maxpool(module):
#         #     for name, child in module.named_children():
#         #         if isinstance(child, nn.MaxPool2d):
#         #             setattr(module, name, nn.Identity())  # 用 nn.Identity() 代替 MaxPool2d
#         #         else:
#         #             remove_maxpool(child)  # 递归替换子模块
        
#         # remove_maxpool(resnet)  # 递归移除所有 MaxPool2d
        
#         # 移除全连接层
#         self.f = nn.Sequential(*list(resnet.children())[:-1])

#     def forward(self, x):
#         x = self.f(x)
#         feature = torch.flatten(x, start_dim=1)
#         return feature

# # 原始版本，权重被重置了，不能调用预训练权重了。
# class Model_base(nn.Module):
#     def __init__(self, input_dim):
#         super(Model_base, self).__init__()

#         self.f = []
#         for name, module in resnet18().named_children():      # 512
#         # for name, module in resnet18().named_children():      # 512
#         # for name, module in resnet50().named_children():         # 2048
#         # for name, module in resnext50_32x4d().named_children():
#         # for name, module in wide_resnet50_2().named_children():
#             if name == 'conv1':
#                 module = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=True)
#             if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
#                 self.f.append(module)
        
#         # encoder
#         self.f = nn.Sequential(*self.f)

#     def forward(self, x):
#         x = self.f(x)
#         # print("forward", x.shape)
#         feature = torch.flatten(x, start_dim=1)
#         return feature
    

class DINOHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(in_dim, 256, bias=False), 
                               nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), 
                                nn.Linear(256, out_dim, bias=True))

    def forward(self, x):
        x = self.g(x)
        return x


class MLP_head(nn.Module):
    def __init__(self, in_dim=512, class_num=16):
        super(MLP_head, self).__init__()
        self.c = nn.Sequential(nn.Linear(in_dim, 256, bias=False), 
                               nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True), 
                               nn.Linear(256, class_num, bias=True))   #2048
    def forward(self, x):
        x = self.c(x)
        return x


class FDGC_head(nn.Module):
    def __init__(self, in_dim=128, class_num=16):
        super(FDGC_head, self).__init__()
        self.c = nn.Sequential(nn.Linear(in_dim, 1024),
                               nn.Dropout(0.5),
                               nn.BatchNorm1d(1024),
                            #    nn.ReLU(inplace=True), 
                               nn.Linear(1024, 256),
                               nn.BatchNorm1d(256),
                            #    nn.ReLU(inplace=True), 
                               nn.Linear(256, class_num)   
                               )   #2048
    def forward(self, x):
        x = self.c(x)
        return x


# print(model)
if __name__ == "__main__":
    input1 = torch.rand(128, 32, 28, 28)
    model = Model_base(32)
    # print(model)
    model2 = FDGC_head(in_dim = 512)

    feature = model(input1)
    print ("feature1", feature.shape)
    out = model2(feature)
    print("output1", out.shape)



