'''
Re-implementation for paper "More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classification"
The official tensorflow implementation is in https://github.com/danfenghong/IEEE_TGRS_MDL-RS
'''

import torch.nn as nn
import torch

class Early_fusion_CNN(nn.Module):
    # Re-implemented early_fusion_CNN for paper "More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    # But not use APs to convert 1-band LiDAR data to 21-band.
    def __init__(self, input_channels, input_channels2, n_classes):
        super(Early_fusion_CNN, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)    # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # For concatenated image x (7×7×d)
        self.conv1 = nn.Conv2d(input_channels + input_channels2, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2 = nn.BatchNorm2d(filters[1])
        # Max pooling ('SAME' mode) --> 4×4×32
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.conv4 = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4 = nn.BatchNorm2d(filters[3])
        # Max pooling ('SAME' mode) --> 2×2×128
        self.conv5 = nn.Conv2d(filters[3], filters[3], (1, 1))
        self.bn5 = nn.BatchNorm2d(filters[3])
        self.conv6 = nn.Conv2d(filters[3], filters[2], (1, 1))
        self.bn6 = nn.BatchNorm2d(filters[2])
        # Average Pooling --> 1×1×64    # Use AdaptiveAvgPool2d() for more robust

        self.conv7 = nn.Conv2d(filters[2], n_classes, (1, 1))

        # weight_init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x1 = x1.squeeze()
        x2 = x2.squeeze()
        # for image x1, x2
        x = self.activation(self.bn1(self.conv1(torch.cat([x1, x2], 1))))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.max_pool(x)
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.max_pool(x)

        x = self.activation(self.bn5(self.conv5(x)))
        x = self.activation(self.bn6(self.conv6(x)))
        x = self.avg_pool(x)
        x = self.conv7(x)

        x = torch.squeeze(x)  # For fully convolutional NN
        return x

class Middle_fusion_CNN(nn.Module):
    # Re-implemented middle_fusion_CNN for paper "More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    # But not use APs to convert 1-band LiDAR data to 21-band.
    def __init__(self, input_channels, input_channels2, n_classes):
        super(Middle_fusion_CNN, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)    # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # For image a (7×7×d)
        self.conv1_a = nn.Conv2d(input_channels, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1_a = nn.BatchNorm2d(filters[0])
        self.conv2_a = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_a = nn.BatchNorm2d(filters[1])
        # Max pooling ('SAME' mode) --> 4×4×32
        self.conv3_a = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3_a = nn.BatchNorm2d(filters[2])
        self.conv4_a = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4_a = nn.BatchNorm2d(filters[3])
        # Max pooling ('SAME' mode) --> 2×2×128

        # For image b (7×7×d)
        self.conv1_b = nn.Conv2d(input_channels2, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1_b = nn.BatchNorm2d(filters[0])
        self.conv2_b = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_b = nn.BatchNorm2d(filters[1])
        # Max pooling ('SAME' mode) --> 4×4×32
        self.conv3_b = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3_b = nn.BatchNorm2d(filters[2])
        self.conv4_b = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4_b = nn.BatchNorm2d(filters[3])
        # Max pooling ('SAME' mode) --> 2×2×128

        self.conv5 = nn.Conv2d(filters[3] + filters[3], filters[3], (1, 1))
        self.bn5 = nn.BatchNorm2d(filters[3])
        self.conv6 = nn.Conv2d(filters[3], filters[2], (1, 1))
        self.bn6 = nn.BatchNorm2d(filters[2])
        # Average Pooling --> 1×1×64    # Use AdaptiveAvgPool2d() for more robust

        self.conv7 = nn.Conv2d(filters[2], n_classes, (1, 1))

        # weight_init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # x1 = x1.squeeze()
        # x2 = x2.squeeze()
        # for image a
        x1 = self.activation(self.bn1_a(self.conv1_a(x1)))
        x1 = self.activation(self.bn2_a(self.conv2_a(x1)))
        x1 = self.max_pool(x1)
        x1 = self.activation(self.bn3_a(self.conv3_a(x1)))
        x1 = self.activation(self.bn4_a(self.conv4_a(x1)))
        x1 = self.max_pool(x1)

        # for image b
        x2 = self.activation(self.bn1_b(self.conv1_b(x2)))
        x2 = self.activation(self.bn2_b(self.conv2_b(x2)))
        x2 = self.max_pool(x2)
        x2 = self.activation(self.bn3_b(self.conv3_b(x2)))
        x2 = self.activation(self.bn4_b(self.conv4_b(x2)))
        x2 = self.max_pool(x2)

        x = self.activation(self.bn5(self.conv5(torch.cat([x1, x2], 1))))
        x = self.activation(self.bn6(self.conv6(x)))
        x = self.avg_pool(x)
        x = self.conv7(x)

        x = torch.squeeze(x)  # For fully convolutional NN
        return x

class Late_fusion_CNN(nn.Module):
    # Re-implemented late_fusion_CNN for paper "More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    # But not use APs to convert 1-band LiDAR data to 21-band.
    def __init__(self, input_channels, input_channels2, n_classes):
        super(Late_fusion_CNN, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)    # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # For image a (7×7×d)
        self.conv1_a = nn.Conv2d(input_channels, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1_a = nn.BatchNorm2d(filters[0])
        self.conv2_a = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_a = nn.BatchNorm2d(filters[1])
        # Max pooling ('SAME' mode) --> 4×4×32
        self.conv3_a = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3_a = nn.BatchNorm2d(filters[2])
        self.conv4_a = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4_a = nn.BatchNorm2d(filters[3])
        # Max pooling ('SAME' mode) --> 2×2×128
        self.conv5_a = nn.Conv2d(filters[3], filters[3], (1, 1))
        self.bn5_a = nn.BatchNorm2d(filters[3])
        self.conv6_a = nn.Conv2d(filters[3], filters[2], (1, 1))
        self.bn6_a = nn.BatchNorm2d(filters[2])
        # Average Pooling --> 1×1×64    # Use AdaptiveAvgPool2d() for more robust

        # For image b (7×7×d)
        self.conv1_b = nn.Conv2d(input_channels2, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1_b = nn.BatchNorm2d(filters[0])
        self.conv2_b = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_b = nn.BatchNorm2d(filters[1])
        # Max pooling ('SAME' mode) --> 4×4×32
        self.conv3_b = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3_b = nn.BatchNorm2d(filters[2])
        self.conv4_b = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4_b = nn.BatchNorm2d(filters[3])
        # Max pooling ('SAME' mode) --> 2×2×128
        self.conv5_b = nn.Conv2d(filters[3], filters[3], (1, 1))
        self.bn5_b = nn.BatchNorm2d(filters[3])
        self.conv6_b = nn.Conv2d(filters[3], filters[2], (1, 1))
        self.bn6_b = nn.BatchNorm2d(filters[2])
        # Average Pooling --> 1×1×64    # Use AdaptiveAvgPool2d() for more robust

        self.conv7 = nn.Conv2d(filters[2] + filters[2], n_classes, (1, 1))

        # weight_init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # x1 = x1.squeeze()
        # x2 = x2.squeeze()
        # for image a
        x1 = self.activation(self.bn1_a(self.conv1_a(x1)))
        x1 = self.activation(self.bn2_a(self.conv2_a(x1)))
        x1 = self.max_pool(x1)
        x1 = self.activation(self.bn3_a(self.conv3_a(x1)))
        x1 = self.activation(self.bn4_a(self.conv4_a(x1)))
        x1 = self.max_pool(x1)
        x1 = self.activation(self.bn5_a(self.conv5_a(x1)))
        x1 = self.activation(self.bn6_a(self.conv6_a(x1)))
        x1 = self.avg_pool(x1)

        # for image b
        x2 = self.activation(self.bn1_b(self.conv1_b(x2)))
        x2 = self.activation(self.bn2_b(self.conv2_b(x2)))
        x2 = self.max_pool(x2)
        x2 = self.activation(self.bn3_b(self.conv3_b(x2)))
        x2 = self.activation(self.bn4_b(self.conv4_b(x2)))
        x2 = self.max_pool(x2)
        x2 = self.activation(self.bn5_b(self.conv5_b(x2)))
        x2 = self.activation(self.bn6_b(self.conv6_b(x2)))
        x2 = self.avg_pool(x2)

        x = self.conv7(torch.cat([x1, x2], 1))

        x = torch.squeeze(x)  # For fully convolutional NN
        return x



class En_De_fusion_CNN(nn.Module):
    def __init__(self, input_channels, input_channels2, out_num):
        super(En_De_fusion_CNN, self).__init__()
        #encoder_layer_1
        self.conv1_x1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16, momentum=0.9),
            nn.ReLU())
        
        self.conv1_x2 = nn.Sequential(
            nn.Conv2d(input_channels2, 16, 3, padding=1),
            nn.BatchNorm2d(16, momentum=0.9),
            nn.ReLU())
        
        #encoder_layer_2
        self.conv2_x1 = nn.Sequential(
            nn.Conv2d(16, 32, 1),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.MaxPool2d(2, 2, padding=1),
            nn.ReLU())

        self.conv2_x2 = nn.Sequential(
            nn.Conv2d(16, 32, 1),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.MaxPool2d(2, 2, padding=1),
            nn.ReLU())
        
        #encoder_layer_3
        self.conv3_x1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU())

        self.conv3_x2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU())

        #encoder_layer_4
        self.conv4_x1 = nn.Sequential(
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.MaxPool2d(2, 2, padding=1),
            nn.ReLU())

        self.conv4_x2 = nn.Sequential(
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.MaxPool2d(2, 2, padding=1),
            nn.ReLU()) 

        #encoder_layer_5
        self.conv5_x1 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128, momentum=0.9), 
            nn.ReLU())    

        #encoder_layer_6
        self.conv6_x1 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.AvgPool2d(2,2), 
            nn.ReLU())

        # fusion_layer   
        self.joint_encoder = nn.Conv2d(64, out_num, 1)

        # decoder_layer
        self.deconv1_1 = nn.ConvTranspose2d(128, 64, 1)
        self.sigmoid = nn.Sigmoid()
        self.deconv1_2 = nn.ConvTranspose2d(128, 64, 1)

        self.deconv2_1 = nn.ConvTranspose2d(64, 32,  3)
        self.deconv2_2 = nn.ConvTranspose2d(64, 32, 3)

        self.deconv3_1 = nn.ConvTranspose2d(32, 16, 3, 2, padding=2)
        self.deconv3_2 = nn.ConvTranspose2d(32, 16, 3, 2, padding=2)
        
        self.deconv4_1 = nn.ConvTranspose2d(16, input_channels, 3, padding=1)
        self.deconv4_2 = nn.ConvTranspose2d(16, input_channels2, 3, padding=1)

    def forward(self, x1, x2):
        # x1 = x1.squeeze()
        # x2 = x2.squeeze()
        
        x1_1 = self.conv1_x1(x1)  
        x2_1 = self.conv1_x2(x2)  
        x1_2 = self.conv2_x1(x1_1)  
        x2_2 = self.conv2_x2(x2_1)  
        x1_3 = self.conv3_x1(x1_2)
        x2_3 = self.conv3_x2(x2_2) 
        x1_4 = self.conv4_x1(x1_3)
        x2_4 = self.conv4_x2(x2_3) 
        joint_layer = torch.cat([x1_4, x2_4], dim=1)
        x1_5 = self.conv5_x1(joint_layer) 
        x1_6 = self.conv6_x1(x1_5) 

        fusion_layer = self.joint_encoder(x1_6)  # [batch, num_classes, 1, 1]
        # print(fusion_layer.shape)
        fusion1_layer = fusion_layer.view(fusion_layer.size(0), -1)

        x_de1_1 = self.deconv1_1(x1_5)
        x_de1_1 = self.sigmoid(x_de1_1)      

        x_de1_2 = self.deconv2_1(x_de1_1)
        x_de1_2 = self.sigmoid(x_de1_2)

        x_de1_3 = self.deconv3_1(x_de1_2)
        x_de1_3 = self.sigmoid(x_de1_3)        

        x_de1_4 = self.deconv4_1(x_de1_3)
        x_de1_4 = self.sigmoid(x_de1_4)     
        
        x_de2_1 = self.deconv1_2(x1_5)
        x_de2_1 = self.sigmoid(x_de2_1)  

        x_de2_2 = self.deconv2_2(x_de2_1)
        x_de2_2 = self.sigmoid(x_de2_2)

        x_de2_3 = self.deconv3_2(x_de2_2)
        x_de2_3 = self.sigmoid(x_de2_3)  

        x_de2_4 = self.deconv4_2(x_de2_3)
        x_de2_4 = self.sigmoid(x_de2_4) 

        return fusion1_layer, x_de1_4, x_de2_4


class Cross_fusion_CNN(nn.Module):
    # Re-implemented Cross_fusion_CNN for paper "More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    # But not use APs to convert 1-band LiDAR data to 21-band.
    def __init__(self, input_channels, input_channels2, n_classes):
        super(Cross_fusion_CNN, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)    # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # For image a (7×7×d)
        self.conv1_a = nn.Conv2d(input_channels, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1_a = nn.BatchNorm2d(filters[0])
        self.conv2_a = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_a = nn.BatchNorm2d(filters[1])
        # Max pooling ('SAME' mode) --> 4×4×32
        self.conv3_a = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3_a = nn.BatchNorm2d(filters[2])
        self.conv4_a = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4_a = nn.BatchNorm2d(filters[3])
        # Max pooling ('SAME' mode) --> 2×2×128

        # For image b (7×7×d)
        self.conv1_b = nn.Conv2d(input_channels2, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1_b = nn.BatchNorm2d(filters[0])
        self.conv2_b = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_b = nn.BatchNorm2d(filters[1])
        # Max pooling ('SAME' mode) --> 4×4×32
        self.conv3_b = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3_b = nn.BatchNorm2d(filters[2])
        self.conv4_b = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4_b = nn.BatchNorm2d(filters[3])
        # Max pooling ('SAME' mode) --> 2×2×128

        self.conv5 = nn.Conv2d(filters[3] + filters[3], filters[3], (1, 1))
        self.bn5 = nn.BatchNorm2d(filters[3])
        self.conv6 = nn.Conv2d(filters[3], filters[2], (1, 1))
        self.bn6 = nn.BatchNorm2d(filters[2])
        # Average Pooling --> 1×1×64    # Use AdaptiveAvgPool2d() for more robust

        self.conv7 = nn.Conv2d(filters[2], n_classes, (1, 1))

        # weight_init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # x1 = x1.squeeze()
        # x2 = x2.squeeze()
        # for image a
        x1 = self.activation(self.bn1_a(self.conv1_a(x1)))
        x1 = self.activation(self.bn2_a(self.conv2_a(x1)))
        x1 = self.max_pool(x1)
        x1 = self.activation(self.bn3_a(self.conv3_a(x1)))


        # for image b
        x2 = self.activation(self.bn1_b(self.conv1_b(x2)))
        x2 = self.activation(self.bn2_b(self.conv2_b(x2)))
        x2 = self.max_pool(x2)
        x2 = self.activation(self.bn3_b(self.conv3_b(x2)))

        x11 = self.activation(self.bn4_a(self.conv4_a(x1)))
        x11 = self.max_pool(x11)
        x22 = self.activation(self.bn4_b(self.conv4_b(x2)))
        x22 = self.max_pool(x22)
        x12 = self.activation(self.bn4_b(self.conv4_b(x1)))
        x12 = self.max_pool(x12)
        x21 = self.activation(self.bn4_a(self.conv4_a(x2)))
        x21 = self.max_pool(x21)

        joint_encoder_layer1 = torch.cat([x11 + x21, x22 + x12], 1)
        joint_encoder_layer2 = torch.cat([x11, x12], 1)
        joint_encoder_layer3 = torch.cat([x22, x21], 1)

        fusion1 = self.activation(self.bn5(self.conv5(joint_encoder_layer1)))
        fusion1 = self.activation(self.bn6(self.conv6(fusion1)))
        fusion1 = self.avg_pool(fusion1)
        fusion1 = self.conv7(fusion1)

        fusion2 = self.activation(self.bn5(self.conv5(joint_encoder_layer2)))
        fusion2 = self.activation(self.bn6(self.conv6(fusion2)))
        fusion2 = self.avg_pool(fusion2)
        fusion2 = self.conv7(fusion2)

        fusion3 = self.activation(self.bn5(self.conv5(joint_encoder_layer3)))
        fusion3 = self.activation(self.bn6(self.conv6(fusion3)))
        fusion3 = self.avg_pool(fusion3)
        fusion3 = self.conv7(fusion3)

        fusion1 = torch.squeeze(fusion1)  # For fully convolutional NN
        fusion2 = torch.squeeze(fusion2)  # For fully convolutional NN
        fusion3 = torch.squeeze(fusion3)  # For fully convolutional NN
        return fusion1, fusion2, fusion3


if __name__ == '__main__':
    l1 = 193
    # l2 = 21
    l2 = 1
    out_num = 2
    
    # x1 = torch.randn(2, 144, 8, 8).cuda()
    # x2 = torch.randn(2, 1, 8, 8).cuda()
    x1 = torch.randn(128, 193, 7, 7).cuda()
    x2 = torch.randn(128, 1, 7, 7).cuda()
    # 创建模型并将其移动到GPU
    net = Middle_fusion_CNN(l1, l2, out_num).cuda()
    net = Late_fusion_CNN(l1, l2, out_num).cuda()
    # net = En_De_fusion_CNN(l1, l2, out_num).cuda()
    # net = Cross_fusion_CNN(l1, l2, out_num).cuda()
    print(net)

    # 运行前向传播
    # y1, y2, y3 = net(x1, x2)
    y1 = net(x1, x2)
    print("Output 1 shape:", y1.shape)
    # print("Output 2 shape:", y2.shape)
    # print("Output 3 shape:", y3.shape)