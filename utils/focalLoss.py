import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_weight_calculation(TrainLabel, args=None):


    # # come from MIVIT
    if args.backbone in ['MIViT']:
        Label = torch.from_numpy(TrainLabel)
        max_class = int(Label.max()) + 1
        loss_weight = torch.ones(max_class)
        sum_num = 0

        for i in range(Label.max()+1):
            loss_weight[i] = len(torch.where(Label == i)[0])
            sum_num = sum_num + len(torch.where(Label == i)[0])

        sum_mean = sum_num / max_class
        weight_out = (sum_mean - loss_weight) / loss_weight

        # 将小于 1 的权重设为 1
        weight_out[torch.where(weight_out < 1)] = 1
        # print("weight_out", weight_out)


    # come from DBCTNet
    elif args.backbone in ['DBCTNet']:
        classes = int(np.max(TrainLabel))+1
        total_w = len(TrainLabel)/classes
        class_map = Counter(TrainLabel) 
        weight_out = [total_w / class_map[i] for i in range(classes)]

        weight_out = [max(w, 1.0) for w in weight_out]  # 下限保护 
        # print("weight_out", weight_out)

    else:
        weight_out = None

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


# come from MIVIT
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
        

# come from DBCTNet
class FocalLoss2(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=3, use_alpha=False, size_average=True):
        super(FocalLoss2, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            #self.alpha = (torch.tensor(alpha)).view(-1,1).cuda()
            self.alpha = torch.tensor(alpha).view(-1,1).cuda()
        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0) #0.0001
        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)
        
        if self.use_alpha:
            alpha = self.alpha[target]
            #alpha = alpha.cuda()
            batch_loss = - alpha.double() * torch.pow(1-prob,self.gamma).double() * (prob.log()).double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        
        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss