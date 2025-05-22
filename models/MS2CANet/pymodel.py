import math
import torch
import torch.nn as nn
# from torchsummary import summary

from .CrosAttention import SA, SE


# ECA ?
class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v1 = self.max_pool(x)
        v = v + v1
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        self.beta = 0.2

    def forward(self, x):
        b, c, h, w = x.size()
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        # y = torch.ones(size=(b,c,h,w), dtype=torch.float32).cuda()
        z = torch.zeros(size=(b, c, h, w), dtype=torch.float32).cuda()

        # change the value of beta to acquire best results
        out = torch.where(out.data>=self.beta, out, z)
        # print(out.grad)

        return out


class EFR(nn.Module):
    def __init__(self, channel):
        super(EFR, self).__init__()
        self.eca = ECABlock(channel)
        self.spatial_attention = SpatialAttentionModule()
        
    def forward(self, x):
        out = self.eca(x)
        out = self.spatial_attention(out) * out
        return out



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)



class PyConv(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[1, 3, 5], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv, self).__init__()

        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], \
                            padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], \
                            padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[2], \
                            padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])
        
        self.channelspatialselayer1 = EFR(channel=64)
        self.channelspatialselayer2 = EFR(channel=64)
        self.channelspatialselayer3 = EFR(channel=128)

    def forward(self, x):
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x1 = self.channelspatialselayer1(x1)
        x2 = self.channelspatialselayer2(x2)
        x3 = self.channelspatialselayer3(x3)
        return torch.cat((x1, x2, x3), dim=1)


def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
    return PyConv(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)



# 特征图数量
class pyCNN(nn.Module):
    def __init__(self, input_channels, input_channels2, classes, FM=32, para_tune=True):
        super(pyCNN, self).__init__()

        self.sa = SA(in_channels = FM * 2)
        self.se = SE(in_channels = FM * 2, para_tune=para_tune)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels = FM, \
                      kernel_size = 3, stride = 1,padding = 1),
            nn.BatchNorm2d(FM),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM, FM * 2, 3, 1, 1),
            nn.BatchNorm2d(FM * 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            get_pyconv(inplans=FM * 2, planes=FM*4, \
                       pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]),
            nn.BatchNorm2d(FM*4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(input_channels2, FM, 3, 1, 1,),
            nn.BatchNorm2d(FM),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(FM, FM*2, 3, 1, 1),
            nn.BatchNorm2d(FM*2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv6 = nn.Sequential(
            get_pyconv(inplans=FM * 2, planes=FM * 4, \
                       pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]),
            nn.BatchNorm2d(FM * 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )
        # self.FusionLayer = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=FM*8,
        #         out_channels=FM*4,
        #         kernel_size=1,
        #     ),
        #     nn.BatchNorm2d(FM*4),
        #     nn.LeakyReLU(),
        # )
        self.out1 = nn.Linear(FM * 4, classes)
        self.out2 = nn.Linear(FM * 4, classes)
        self.out3 = nn.Linear(FM * 4, classes)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(FM * 4, classes)
    
    def forward(self, x1, x2):
        # print("input x1", x1.shape, "input x2", x2.shape)    # [64, 20, 11, 11] / [64, 1, 11, 11]

        x1 = self.conv1(x1)
        x2 = self.conv4(x2)
        # print("layer1 x1", x1.shape, "layer1 x2", x2.shape)  # [64, 64, 5, 5] / [64, 64, 5, 5]

        x1 = self.conv2(x1)
        x2 = self.conv5(x2)
        # print("layer2 x1", x1.shape, "layer2 x2", x2.shape)  #  [64, 128, 2, 2] / [64, 128, 2, 2]

        x1 = self.sa(x1, x2)
        x2 = self.se(x2, x1)
        # print("sa x1", x1.shape, "se x2", x2.shape)          #  [64, 128, 2, 2] / [64, 128, 2, 2]

        x1 = self.conv3(x1)
        x2 = self.conv6(x2)
        # print("layer3 x1", x1.shape, "layer3 x2", x2.shape)          # [64, 256, 1, 1] / [64, 256, 1, 1]

        x1 = x1.view(x1.size(0), -1)  # flatten the output of conv2 
        out1 = self.out1(x1)

        x2 = x2.view(x2.size(0), -1)
        out2 = self.out2(x2)
        # print("out1", out1.shape, "out2", out2.shape) # 64, 15
        x = x1 + x2

        out3 = self.out3(x)
        return out1, out2, out3


# cnn = pyCNN(NC=40, classes=13,FM=64)
# a = torch.randn(size=(64,40,11,11))
# b = torch.randn(size=(64,1,11,11))
#
# c,d,e = cnn(a,b)
# print(c)
# print()

# py = get_pyconv(inplans=128,planes=256,pyconv_kernels=[3,5,7],stride=1,pyconv_groups=[1,4,8])
# a = torch.randn(size=(64,128,2,2))
# b = py(a)

# pe = ChannelSpatialSELayer2D(num_channels=64,reduction_ratio=2)
# sp = SpatialSELayer2D(num_channels=64)
#
# a = torch.randn(size=(64,64,11,11))
# b = pe(a)
# c = sp(a)
# print(b)

# print()
