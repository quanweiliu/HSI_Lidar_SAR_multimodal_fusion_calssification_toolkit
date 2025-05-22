
import math
import torch
import torch.nn as nn

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
        # print("out", out.shape)

        out = self.sigmoid(self.conv2d(out))
        # y = torch.ones(size=(b,c,h,w), dtype=torch.float32).cuda()
        z = torch.zeros(size=(b, c, h, w), dtype=torch.float32).cuda()
        # beta = 0.2
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
    

if __name__ == "__main__":

    a = torch.randn(size=(64, 64, 11, 11)).cuda()
    model = EFR(channel = 64).cuda()
    c = model(a)
    print("c1", c.shape)

    model = ECABlock(channels = 64).cuda()
    c = model(a)
    print("c2", c.shape)

    model = SpatialAttentionModule().cuda()
    c = model(a)
    print("c3", c.shape)