import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# source from https://github.com/Yejin0111/ADD-GCN
# 动态图
class DGraph(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(DGraph, self).__init__()

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward_construct_dynamic_graph(self, x):
        m_batchsize, channel, class_num = x.size()
        proj_query = x
        proj_key = x.view(m_batchsize, channel, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, channel, -1)
        out = torch.bmm(attention, proj_value) 
        x_glb = self.gamma*out + x
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)                  
        dynamic_adj = torch.sigmoid(dynamic_adj)                  
        return dynamic_adj


    def forward(self, x):
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x      


class SGraph(nn.Module):
# 静态图
    def __init__(self, in_features, out_features, num_nodes):
        super(SGraph, self).__init__()

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            gamma = self._compute_gamma(x)
            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, 
                            stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1


class Branch1(nn.Module):
    def __init__(self, input_channels, feature=64, 
                num_node=4, drop_prob=0.1, block_size=3):
        super(Branch1, self).__init__()
        self.input_channels = input_channels
        self.num_node = num_node
        self.feature = feature
        self.dropblock = LinearScheduler(
                            DropBlock2D(drop_prob=drop_prob, \
                                        block_size=block_size),
                                        start_value=0.,
                                        stop_value=drop_prob,
                                        nr_steps=5e3)
        # bone 
        #TODO add 1*1 conv layer to reduce the channel
        self.conv0 = nn.Conv2d(self.input_channels, 32, 1)

        self.conv1 = nn.Conv2d(self.input_channels, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2))
        # get size
        self.features_size = self._get_final_flattened_size()
        # get adjancency matrix
        self.fc_sam = nn.Conv2d(64, self.num_node, (1,1), bias=False)
        self.conv_transform = nn.Conv2d(64, feature, (1,1))
        # graph model
        self.sgraph = SGraph(feature, feature, self.num_node)
        self.dgraph = DGraph(feature, feature, self.num_node)
        

    def _get_final_flattened_size(self):
        with torch.no_grad():
            pass
        return self.feature*self.num_node
    
    def forward_sam(self, x):
        mask = self.fc_sam(x)
        mask = mask.view(mask.size(0), mask.size(1), -1) 
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)
        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x

    def forward(self, x):
        # x = x.squeeze()
        self.dropblock.step() 
        # print("x", x.shape)  # torch.Size([B, 64, 5, 5])

        x1 = F.leaky_relu(self.conv1(x))
        x1 = self.bn1(x1)
        x1 = self.dropblock(x1)
        # print("x1", x1.shape)  # torch.Size([B, 64, 5, 5])

        x2 = F.leaky_relu(self.conv2(x1))
        x2 = self.bn2(x2)
        x2 = self.dropblock(x2)
        # print("x2", x2.shape)  # torch.Size([B, 64, 3, 3])

        x3 = self.forward_sam(x2) 
        # print("x3", x3.shape)  # torch.Size([B, 64, 60])
        x4 = self.sgraph(x3) + x3
        # print("x4", x4.shape)  # torch.Size([B, 64, 60])
        x5 = self.dgraph(x4) + x4
        # print("x5", x5.shape)  # torch.Size([B, 64, 60])
        x6 = x5.view(-1, x5.size(1) * x5.size(2))
        # print("x6", x6.shape)  # torch.Size([B, 3840])
        return x6, x2
    

class Branch2(nn.Module):
    def __init__(self, input_channels, feature=64, 
                num_node=4, drop_prob=0.1, block_size=3):
        super(Branch2, self).__init__()
        self.input_channels = input_channels
        self.num_node = num_node
        self.feature = feature
        self.dropblock = LinearScheduler(
                            DropBlock2D(drop_prob=drop_prob, \
                                        block_size=block_size),
                                        start_value=0.,
                                        stop_value=drop_prob,
                                        nr_steps=5e3)
        # bone 
        self.conv1 = nn.Conv2d(self.input_channels, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2))
        # get size
        self.features_size = self._get_final_flattened_size()
        # get adjancency matrix
        self.fc_sam = nn.Conv2d(64, self.num_node, (1,1), bias=False)
        self.conv_transform = nn.Conv2d(64, feature, (1,1))
        # graph model
        self.sgraph = SGraph(feature, feature, self.num_node)
        self.dgraph = DGraph(feature, feature, self.num_node)
        

    def _get_final_flattened_size(self):
        with torch.no_grad():
            pass
        return self.feature*self.num_node
    
    def forward_sam(self, x):
        mask = self.fc_sam(x)
        mask = mask.view(mask.size(0), mask.size(1), -1) 
        mask = torch.sigmoid(mask).transpose(1, 2)
        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x

    def forward(self, x):
        # x = x.squeeze()
        self.dropblock.step() 
        # print("x", x.shape)  # torch.Size([B, 64, 5, 5])

        x1 = F.leaky_relu(self.conv1(x))
        x1 = self.bn1(x1)
        x1 = self.dropblock(x1)
        # print("x1", x1.shape)  # torch.Size([B, 64, 5, 5])

        x2 = F.leaky_relu(self.conv2(x1))
        x2 = self.bn2(x2)
        x2 = self.dropblock(x2)
        # print("x2", x2.shape)  # torch.Size([B, 64, 3, 3])

        x3 = self.forward_sam(x2) 
        # print("x3", x3.shape)  # torch.Size([B, 64, 60])
        x4 = self.sgraph(x3) + x3
        # print("x4", x4.shape)  # torch.Size([B, 64, 60])
        x5 = self.dgraph(x4) + x4
        # print("x5", x5.shape)  # torch.Size([B, 64, 60])
        x6 = x5.view(-1, x5.size(1) * x5.size(2))
        # print("x6", x6.shape)  # torch.Size([B, 3840])
        return x6, x2
    

class SHNet(nn.Module):
    def __init__(self, l1, l2, num_classes, feature=64, factors=8):
        super(SHNet, self).__init__()
        self.num_classes = num_classes
        self.feature = feature
        self.factors = factors
        self.num_node = num_classes*factors

        self.branch1 = Branch1(l1, feature=feature, num_node=self.num_node)
        self.branch2 = Branch2(l2, feature=feature, num_node=self.num_node)

        # last
        print("self._get_final_flattened_size()", self._get_final_flattened_size())
        self.fc1 = nn.Linear(self._get_final_flattened_size(), 1024)
        self.drop1 = nn.Dropout(0.5)
        self.bn_f1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn_f2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)    

    def _get_final_flattened_size(self):
        with torch.no_grad():
            pass
        return self.feature*self.num_classes*self.factors*2
    
    def get_visulization(self, x1, x2):
        x1, v1 = self.branch1(x1)
        x2, v2 = self.branch2(x2)
        return v1, v2

    def forward(self, x1, x2):
        x1, v1 = self.branch1(x1)
        x2, v2 = self.branch2(x2)
        # print("x1", x1.shape, "x2", x2.shape)   # 20, 2048

        joint_layer = torch.cat([x1, x2], dim=1)

        # return joint_layer
        # print("joint_layer", joint_layer.shape)
        x = F.leaky_relu(self.fc1(joint_layer))
        x = self.bn_f1(x)
        x = self.drop1(x)
        # print("x", x.shape)

        x = F.leaky_relu(self.fc2(x))
        x = self.bn_f2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    num_classes = 15
    factors = 2
    feature = 16
    hsi_bands = 144
    sar_bands = 1
    x1 = torch.randn(8, hsi_bands, 7, 7)
    x2 = torch.randn(8, sar_bands, 7, 7)
    model = SHNet(l1=hsi_bands, l2=sar_bands, \
                  num_classes=num_classes, feature=feature, factors=factors)
    # print(model)
    output = model(x1, x2)
    print(output.shape)
