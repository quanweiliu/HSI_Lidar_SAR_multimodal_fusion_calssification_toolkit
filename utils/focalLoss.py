import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_weight_calculation(TrainLabel):
    TrainLabel = torch.from_numpy(TrainLabel)
    max_class = int(TrainLabel.max()) + 1
    loss_weight = torch.ones(max_class)
    sum_num = 0

    for i in range(TrainLabel.max()+1):
        loss_weight[i] = len(torch.where(TrainLabel == i)[0])
        sum_num = sum_num + len(torch.where(TrainLabel == i)[0])

    sum_mean = sum_num / max_class
    # print(loss_weight)
    # print(sum_num)
    # print(sum_mean)
    weight_out = (sum_mean-loss_weight) / loss_weight

    # 将小于1的权重设为1
    weight_out[torch.where(weight_out < 1)] = 1
    return weight_out  # (1 - loss_weight/sum) / ((1 - loss_weight / sum).sum())

# test 不太需要计算损失，可以直接用这个代替
def loss_weight_calculation_test(TestLabel):

    return torch.ones(int(TestLabel.max())+1)

def loss_weight_calculation_np(TrainLabel):
    max_class = int(TrainLabel.max()) + 1
    loss_weight = np.ones(max_class)
    sum_num = 0

    for i in range(max_class):
        loss_weight[i] = len(np.where(TrainLabel == i)[0])
        sum_num = sum_num + len(np.where(TrainLabel == i)[0])

    # print(loss_weight)
    # print(sum)
    sum_mean = sum_num / max_class
    weight_out = (sum_mean - loss_weight) / loss_weight

    # 将小于1的权重设为1
    weight_out[weight_out < 1] = 1
    return weight_out


class FocalLoss(nn.Module):
    def __init__(self, loss_weight, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight = self.loss_weight, reduction= 'none')
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss