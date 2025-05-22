import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_dim=128, class_num=15):
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


class MS2_head(nn.Module):
    def __init__(self, in_dim=256, class_num=16):
        super(MS2_head, self).__init__()

        self.out1 = nn.Linear(in_dim, class_num)
        self.out2 = nn.Linear(in_dim, class_num)
        self.out3 = nn.Linear(in_dim, class_num)

    def forward(self, x1, x2):
        out1 = self.out1(x1)
        out2 = self.out2(x2)
        x = x1 + x2
        out3 = self.out3(x)
        
        return out1, out2, out3
