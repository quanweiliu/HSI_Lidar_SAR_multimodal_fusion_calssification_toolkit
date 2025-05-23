import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [0]))
print('using GPU %s' % ','.join(map(str, [0])))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from thop import profile, clever_format

import csv
import time
import numpy as np
import json
from datetime import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 

from option import opt
from loadData import data_pipe
from loadData.dataAugmentation import dataAugmentation
from models import CNNs, vision_transformer, mamba, S2ENet, FusAtNet, SHNet, heads
from models import MDL

# from models.MS2CANet import pymodel
from models.MS2CANet2 import pymodel
from models.CrossHL import CrossHL
from models.HCTNet import HCTNet
from models.DSHFNet import DSHF
from models.MIViT import MMA
from models import get_model_config

from utils import trainer, tester, focalLoss


args = opt.get_args()
args.dataset_name = "PaviaU"
# args.dataset_name = "Houston_2013"
# args.dataset_name = "Houston_2018"
# args.dataset_name = "Augsburg"
# args.dataset_name = "Berlin"
# args.dataset_name = "MelasChasma"
# args.dataset_name = "CopratesChasma"
# args.dataset_name = "GaleCrater"


# args.backbone = "vit"
# args.backbone = "cnn"
# args.backbone = "mamba"

# args.backbone = "MDL_M"
# args.backbone = "MDL_L"
# args.backbone = "MDL_E_D"
# args.backbone = "MDL_C"


# args.backbone = "MS2CANet"
# args.backbone = "S2ENet"
# args.backbone = "FusAtNet"
# args.backbone = "CrossHL"
# args.backbone = "HCTNet"
# args.backbone = "DSHFNet"
# args.backbone = "MIViT"  
args.backbone = "SHNet"

args.split_type = "disjoint"
get_model_config(args)

print("args.backbone", args.backbone)
# print("args.randomCrop", args.randomCrop)


# data_pipe.set_deterministic(seed = 666)
args.print_data_info = True
args.data_info_start = 1
args.show_gt = False
args.remove_zero_labels = True


if args.backbone in args.SSISO:
    print("args.randomCrop", args.randomCrop)
    transform = dataAugmentation(args.randomCrop)   # 有些模型加增强，会造成测试精度下降很多
else:
    transform = None


# # create dataloader
# if args.dataset_name in args.SD:
#     img2 = None
#     args.train_ratio = 0.8
#     args.path_data = "/home/icclab/Documents/lqw/DatasetSMD"
#     img1, train_gt, val_gt, data_gt = data_pipe.get_data(args)
#     print(img1.shape, train_gt.shape, val_gt.shape, data_gt.shape)
if args.dataset_name in args.MD:
    args.train_ratio = 0.8
    args.path_data = "/home/icclab/Documents/lqw/DatasetMMF"
    img1, img2, train_gt, val_gt, test_gt, data_gt = data_pipe.get_data(args)
    print(img1.shape, img2.shape, train_gt.shape, test_gt.shape, data_gt.shape)


if args.backbone in args.MMISO or args.backbone in args.MMIMO:
    print("mutlisacle multimodality")
    # 在这直接输出多尺度的图像
    train_dataset = data_pipe.HyperXMM(img1, data2=img2, gt=train_gt, 
                                    transform=transform, patch_size=args.patch_size, 
                                    remove_zero_labels=args.remove_zero_labels)
    val_dataset = data_pipe.HyperXMM(img1, data2=img2, gt=val_gt, 
                                    transform=transform, patch_size=args.patch_size, 
                                    remove_zero_labels=args.remove_zero_labels)
    test_dataset = data_pipe.HyperXMM(img1, data2=img2, gt=test_gt, 
                                    transform=None, patch_size=args.patch_size, 
                                    remove_zero_labels=args.remove_zero_labels)
    
    height, wigth, data1_bands = train_dataset.data1.shape
    height, wigth, data2_bands = train_dataset.data2.shape

    # 用于 focalloss
    train_gt_pure = train_gt[train_gt > 0] - 1
    val_gt_pure = val_gt[val_gt > 0] - 1
    test_gt_pure = test_gt[test_gt > 0] - 1
    loss_weight = focalLoss.loss_weight_calculation(train_gt_pure)
    print("data1", train_dataset.data1.shape, "data2", train_dataset.data2.shape)


elif args.backbone in args.SSISO or args.backbone in args.SMIMO \
    or args.backbone in args.SMISO or args.backbone in args.SMIMO2 \
    or args.backbone in args.SMIMO3:

    print("singlescale multimodality")
    train_dataset = data_pipe.HyperX(img1, data2=img2, gt=train_gt, 
                                    transform=transform, patch_size=args.patch_size, 
                                    remove_zero_labels=args.remove_zero_labels)
    val_dataset = data_pipe.HyperX(img1, data2=img2, gt=val_gt, 
                                    transform=transform, patch_size=args.patch_size, 
                                    remove_zero_labels=args.remove_zero_labels)
    test_dataset = data_pipe.HyperX(img1, data2=img2, gt=test_gt, 
                                    transform=None, patch_size=args.patch_size, 
                                    remove_zero_labels=args.remove_zero_labels)
    
    height, wigth, data1_bands = train_dataset.data1.shape
    height, wigth, data2_bands = train_dataset.data2.shape
    print("data1", train_dataset.data1.shape, "data2", train_dataset.data2.shape)


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

class_num = np.max(train_gt)
print(class_num, train_gt.shape, len(train_loader.dataset))


if args.backbone in args.MMISO or args.backbone in args.MMIMO:
    for data11, data12, data13, data21, data22, data23, label in train_loader:
        print("x.shape, y.shape", data11.shape, data12.shape, data13.shape)
        print("x.shape, y.shape", data11.dtype, data12.dtype, data13.dtype)

        print("x.shape, y.shape", data21.shape, data22.shape, data23.shape)
        print("x.shape, y.shape", data21.dtype, data22.dtype, data23.dtype)
        break
elif args.backbone in args.SMISO \
    or args.backbone in args.SMIMO \
    or args.backbone in args.SMIMO2 \
    or args.backbone in args.SMIMO3:
    
    for data11, data12, label in train_loader:
        print("x.shape, y.shape", data11.shape, data12.shape, label.shape)
        print("x.shape, y.shape", data11.dtype, data12.dtype, label.dtype)
        break
elif args.backbone in args.SSISO:
    for x, z in train_loader:
        print("x.shape, y.shape", x.shape, z.shape)
        break

args.result_dir = os.path.join("/home/icclab/Documents/lqw/Multimodal_Classification/MultiModal/result_Berlin",
                    datetime.now().strftime("%m-%d-%H-%M-") + args.backbone)
print(args.result_dir)

# 加载已有权重路径
# args.result_dir = "/home/liuquanwei/code/DMVL_joint_MNDIS/results_final/08-09-17-05-vit_D8"
if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
with open(args.result_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)


criterion = torch.nn.CrossEntropyLoss()
super_head = None

if args.backbone == "cnn":
    model = CNNs.Model_base(args.components, backbone='resnet18', is_pretrained=True).cuda()
    args.feature_dim = 512
    # args.feature_dim = 2048
    super_head = heads.FDGC_head(args.feature_dim, class_num=class_num).to(args.device)
    params = list(super_head.parameters())  + list(model.parameters())

elif args.backbone == "vit":
    model = vision_transformer.vit_hsi(args.components, args.randomCrop).to(args.device)
    # encoder = vision_transformer.vit_small(args.components, args.randomCrop).to(args.device)
    args.feature_dim = 126
    super_head = heads.FDGC_head(args.feature_dim, class_num=class_num).to(args.device)
    params = list(super_head.parameters())  + list(model.parameters())

elif args.backbone == "mamba":
    model = mamba.Vim(
        dim=64,  # Dimension of the transformer model
        # heads=8,  # Number of attention heads
        dt_rank=32,  # Rank of the dynamic routing matrix
        dim_inner=64,  # Inner dimension of the transformer model
        d_state=64,  # Dimension of the state vector
        num_classes=10,  # Number of output classes
        image_size=args.randomCrop,  # Size of the input image
        patch_size=4,  # Size of each image patch
        channels=args.components,  # Number of input channels
        dropout=0.1,  # Dropout rate
        depth=4,  # Depth of the transformer model
    ).to(args.device)
    args.feature_dim = 64
    super_head = heads.FDGC_head(args.feature_dim, class_num=class_num).to(args.device)
    params = list(super_head.parameters())  + list(model.parameters())


elif args.backbone == "MDL_M":
    model = MDL.Middle_fusion_CNN(data1_bands, data2_bands, class_num).to(args.device)
    params = model.parameters()
    print("model: ", "MDL_M")

elif  args.backbone == "MDL_L":
    model = MDL.Late_fusion_CNN(data1_bands, data2_bands, class_num).to(args.device)
    params = model.parameters()
    print("model: ", "MDL_L")

elif args.backbone == "MDL_E_D":
    model = MDL.En_De_fusion_CNN(data1_bands, data2_bands, class_num).to(args.device)
    params = model.parameters()
    print("model: ", "MDL_E_D")

elif  args.backbone == "MDL_C":
    model = MDL.Cross_fusion_CNN(data1_bands, data2_bands, class_num).to(args.device)
    params = model.parameters()
    print("model: ", "MDL_C")


elif args.backbone == "MS2CANet":
    FM = 64
    args.feature_dim = 256
    para_tune = False
    if args.dataset_name == "Houston_2013":
        para_tune = True                # para_tune 这个参数对于 Houston 的提升有两个点！！

    # model = pymodel.pyCNN(data1_bands, data2_bands, classes=class_num, \
    #                       FM=FM, para_tune=para_tune).to(args.device)
    # params = model.parameters()

    model = pymodel.pyCNN(data1_bands, data2_bands, FM=FM, para_tune=para_tune).to(args.device)
    super_head = heads.MS2_head(args.feature_dim, class_num=class_num).to(args.device)
    params = list(super_head.parameters())  + list(model.parameters())

elif args.backbone == 'S2ENet':
    model = S2ENet.S2ENet(data1_bands, data2_bands, class_num, \
                            patch_size=args.patch_size).to(args.device)
    params = model.parameters()

elif args.backbone == "FusAtNet":
    model = FusAtNet.FusAtNet(data1_bands, data2_bands, class_num).to(args.device)
    params = model.parameters()

elif args.backbone == "CrossHL":
    FM = 16
    model = CrossHL.CrossHL_Transformer(FM, data1_bands, data2_bands, class_num, \
                                        args.patch_size).to(args.device)
    params = model.parameters()

elif args.backbone == "HCTNet":
    model = HCTNet(in_channels=1, num_classes=class_num).to(args.device)
    params = model.parameters()

elif args.backbone == "SHNet":
    FM = 64
    # FM = 16
    model = SHNet.SHNet(data1_bands, data2_bands, feature=FM, \
                        num_classes=class_num, factors=args.factors).to(args.device)
    params = model.parameters()

elif args.backbone == "DSHFNet":
    model = DSHF(l1=data1_bands, l2=data2_bands, \
                num_classes=class_num, encoder_embed_dim=64).to(args.device)
    params = model.parameters()

elif args.backbone == "MIViT":
    model = MMA.MMA(l1=data1_bands, l2=data2_bands, patch_size=args.patch_size, \
                num_patches=64, num_classes=class_num,
                encoder_embed_dim=64, decoder_embed_dim=32, en_depth=5, \
                en_heads=4, de_depth=5, de_heads=4, mlp_dim=8, dropout=0.1, \
                emb_dropout=0.1,fusion=args.fusion).to(args.device)
    params = model.parameters()
    
    loss_weight = loss_weight.to(args.device)
    criterion = focalLoss.FocalLoss(loss_weight, gamma=2, alpha=None)

else:
    raise NotImplementedError("No models")
print("backbone: ", args.backbone)


if not args.schedule:
	print("marker")
	optimizer = optim.Adam(params, lr=args.learning_rate)

elif args.backbone == "S2ENet":
	print("marker2")
	optimizer = optim.Adam(params, lr=args.learning_rate)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 150, 180], gamma=0.1)

else:
	print("marker3")
	optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# args.resume = os.path.join(args.result_dir, "joint_oa_model.pth")
if args.resume != '':
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['base'], strict=False)
    epoch_start = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(args.resume))
else:
    epoch_start = 0


best_loss = 999
best_acc = 0
train_losses = []

total_train_time = time.time()

for epoch in range(epoch_start, args.epochs):
    if args.backbone in args.SSISO:
        train_loss, train_accuracy, test_accuracy, train_time \
                                        = trainer.train_SSISO(epoch, model, super_head, \
                                        criterion, train_loader, val_loader, optimizer, args)
    
    elif args.backbone in args.SMIMO:
        train_loss, train_accuracy, test_accuracy, train_time \
                                        = trainer.train_SMIMO(epoch, model, super_head, \
                                        criterion, train_loader, val_loader, optimizer, args)

    elif args.backbone in args.SMIMO2:
        train_loss, train_accuracy, test_accuracy, train_time \
                                        = trainer.train_SMIMO2(epoch, model, criterion, \
                                        train_loader, val_loader, optimizer, args)
        
    elif args.backbone in args.SMIMO3:
        train_loss, train_accuracy, test_accuracy, train_time \
                                        = trainer.train_SMIMO3(epoch, model, criterion, \
                                        train_loader, val_loader, optimizer, args)

    elif args.backbone in args.SMISO:
        train_loss, train_accuracy, test_accuracy, train_time \
                                        = trainer.train_SMISO(epoch, model, criterion, \
                                        train_loader, val_loader, optimizer, args)
        
    elif args.backbone in args.MMISO:
        train_loss, train_accuracy, test_accuracy, train_time \
                                        = trainer.train_MMISO(epoch, model, criterion, \
                                        train_loader, val_loader, optimizer, args)
        
    elif args.backbone in args.MMIMO:
        train_loss, train_accuracy, test_accuracy, train_time \
                                        = trainer.train_MMIMO(epoch, model, criterion, \
                                        train_loader, val_loader, optimizer, args)
    else:
        raise NotImplementedError("NO this model")
    train_losses.append(train_loss)
    
    if not args.schedule:
        pass
    else:
        scheduler.step()
    
    with open(os.path.join(args.result_dir, "log.csv"), 'a+', encoding='gbk') as f:
        row=[["epoch", epoch, 
            "loss", train_loss, 
            "train_accuracy", round(train_accuracy, 2),
            "test_accuracy", round(test_accuracy, 2),
            "train_time", round(train_time, 2),
            '\n']]
        write=csv.writer(f)
        for i in range(len(row)):
            write.writerow(row[i])

    if train_loss < best_loss:
        best_loss = train_loss
        if super_head != None:
            torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "super_head": super_head.state_dict(),
                    "optimizer": optimizer.state_dict()}, 
                    os.path.join(args.result_dir, "model_loss.pth"))
        else:
            torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, 
                    os.path.join(args.result_dir, "model_loss.pth"))

    if best_acc < test_accuracy:
        best_acc = test_accuracy
        if super_head != None:
            torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "super_head": super_head.state_dict(),
                    "optimizer": optimizer.state_dict()}, 
                    os.path.join(args.result_dir, "model_acc.pth"))
        else:
            torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, 
                    os.path.join(args.result_dir, "model_acc.pth"))
        
total_train_time = time.time() - total_train_time

if super_head != None:
    torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "super_head": super_head.state_dict(),
            "optimizer": optimizer.state_dict()}, 
            os.path.join(args.result_dir, "model_loss.pth"))
else:
    torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()}, 
            os.path.join(args.result_dir, "model_last.pth"))


# args.result_dir = "/home/leo/Multimodal_Classification/MyMultiModal/result/03-25-12-26-MIViT"
args.resume = os.path.join(args.result_dir, "model_acc.pth")
# args.resume = os.path.join(args.result_dir, "model_loss.pth")
if args.resume != '':
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model'], strict=False)
    if super_head != None:
        super_head.load_state_dict(checkpoint['super_head'], strict=False)
    epoch = checkpoint['epoch'] + 1
    print('Loaded from: {} epoch {}'.format(args.resume, epoch))
else:
    epoch_start = 0

tic = time.time()

if args.backbone in args.SSISO:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_SSISO(model, super_head, criterion, test_loader, args)
if args.backbone in args.SMIMO:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_SMIMO(model, super_head, criterion, test_loader, args)
if args.backbone in args.SMIMO2:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_SMIMO2(model, criterion, test_loader, args)
if args.backbone in args.SMIMO3:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_SMIMO3(model, criterion, test_loader, args)
elif args.backbone in args.SMISO:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_SMISO(model, criterion, test_loader, args)
elif args.backbone in args.MMISO:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_MMISO(model, criterion, test_loader, args)
elif args.backbone in args.MMIMO:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_MMIMO(model, criterion, test_loader, args)
classification, kappa = tester.get_results(test_preds, targets)

test_time = time.time() - tic

with open(os.path.join(args.result_dir, "log_final.csv"), 'a+', encoding='gbk') as f:
    row=[["training",
        "\nepoch", epoch, 
        "\ndata_name = " + str(args.dataset_name),
        "\nbatch_size = " + str(args.batch_size),
        "\npatch_size = " + str(args.patch_size),
        "\nnum_components = " + str(args.components),
        '\n' + classification,
        "\nkappa = \t\t\t" + str(round(kappa, 4)),
        "\ntotal_time = ", round(total_train_time, 2),
        '\ntest time = \t' + str(round(test_time, 2)),
        ]]
    write=csv.writer(f)
    for i in range(len(row)):
        write.writerow(row[i])


# args.result_dir = "/home/leo/Multimodal_Classification/MyMultiModal/result/03-25-12-26-MIViT"
# args.resume = os.path.join(args.result_dir, "model_acc.pth")
args.resume = os.path.join(args.result_dir, "model_loss.pth")
if args.resume != '':
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model'], strict=False)
    if super_head != None:
        super_head.load_state_dict(checkpoint['super_head'], strict=False)
    epoch = checkpoint['epoch'] + 1
    print('Loaded from: {} epoch {}'.format(args.resume, epoch))
else:
    epoch_start = 0

tic = time.time()

if args.backbone in args.SSISO:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_SSISO(model, super_head, criterion, test_loader, args)
if args.backbone in args.SMIMO:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_SMIMO(model, super_head, criterion, test_loader, args)
if args.backbone in args.SMIMO2:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_SMIMO2(model, criterion, test_loader, args)
if args.backbone in args.SMIMO3:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_SMIMO3(model, criterion, test_loader, args)
elif args.backbone in args.SMISO:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_SMISO(model, criterion, test_loader, args)
elif args.backbone in args.MMISO:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_MMISO(model, criterion, test_loader, args)
elif args.backbone in args.MMIMO:
    test_losses, test_preds, correct, targets = \
        tester.linear_test_MMIMO(model, criterion, test_loader, args)
classification, kappa = tester.get_results(test_preds, targets)

test_time = time.time() - tic

with open(os.path.join(args.result_dir, "log_final.csv"), 'a+', encoding='gbk') as f:
    row=[["training",
        "\nepoch", epoch, 
        "\ndata_name = " + str(args.dataset_name),
        "\nbatch_size = " + str(args.batch_size),
        "\npatch_size = " + str(args.patch_size),
        "\nnum_components = " + str(args.components),
        '\n' + classification,
        "\nkappa = \t\t\t" + str(round(kappa, 4)),
        "\ntotal_time = ", round(total_train_time, 2),
        '\ntest time = \t' + str(round(test_time, 2)),
        ]]
    write=csv.writer(f)
    for i in range(len(row)):
        write.writerow(row[i])



args.resume = os.path.join(args.result_dir, "model_acc.pth")
# args.resume = os.path.join(args.result_dir, "model_loss.pth")
if args.resume != '':
    checkpoint = torch.load(args.resume)
    # print("checkpoint", checkpoint.keys())
    model.load_state_dict(checkpoint['model'], strict=False)
    if super_head != None:
        super_head.load_state_dict(checkpoint['super_head'], strict=False)
    epoch = checkpoint['epoch'] + 1
    print('Loaded from: {} epoch {}'.format(args.resume, epoch))
else:
    epoch_start = 0


if args.backbone in args.SSISO:
    print("args.randomCrop", args.randomCrop)
    transform = dataAugmentation(args.randomCrop)   # 有些模型加增强，会造成测试精度下降很多
else:
    transform = None


args.print_data_info = False
args.data_info_start = 1
args.show_gt = False
args.remove_zero_labels = False


# create dataloader
if args.dataset_name in args.SD:
    args.train_ratio = 0.1
    args.path_data = "/home/icclab/Documents/lqw/DatasetSMD"
    img1, train_gt, test_gt, GT = data_pipe.get_data(args)
    print(img1.shape, train_gt.shape, test_gt.shape, GT.shape)
elif args.dataset_name in args.MD:
    args.train_ratio = 1
    args.path_data = "/home/icclab/Documents/lqw/DatasetMMF"
    img1, img2, train_gt, val_gt, test_gt, GT = data_pipe.get_data(args)
    print(img1.shape, img2.shape, train_gt.shape, test_gt.shape, GT.shape)

if args.backbone in args.MMISO or args.backbone in args.MMIMO:
    print("mutlisacle multimodality")
    # 在这直接输出多尺度的图像
    data_dataset = data_pipe.HyperXMM(img1, data2=img2, gt=train_gt, 
                                    transform=None, patch_size=args.patch_size, 
                                    remove_zero_labels=args.remove_zero_labels)

    # 用于 focalloss
    # train_gt_pure = train_gt[train_gt > 0] - 1
    # test_gt_pure = test_gt[test_gt > 0] - 1
    # loss_weight = focalLoss.loss_weight_calculation(test_gt_pure)
    print("data1", data_dataset.data1.shape, "data2", data_dataset.data2.shape)


elif args.backbone in args.SSISO or args.backbone in args.SMIMO \
    or args.backbone in args.SMISO or args.backbone in args.SMIMO2 \
    or args.backbone in args.SMIMO3:

    print("singlescale multimodality")
    data_dataset = data_pipe.HyperX(img1, data2=img2, gt=train_gt, 
                                    transform=transform, patch_size=args.patch_size, 
                                    remove_zero_labels=args.remove_zero_labels)
    print("data1", data_dataset.data1.shape, "data2", data_dataset.data2.shape)


data_loader = DataLoader(data_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)


if args.backbone in args.SSISO:
    tester.linear_visualation_SSISO(model, super_head, data_loader, args, groundTruth=GT, visulation=True)
if args.backbone in args.SMIMO:
    tester.linear_visualation_SMIMO(model, super_head, data_loader, args, groundTruth=GT, visulation=True)
if args.backbone in args.SMIMO2:
    tester.linear_visualation_SMIMO2(model, data_loader, args, groundTruth=GT, visulation=True)
if args.backbone in args.SMIMO3:
    tester.linear_visualation_SMIMO3(model, data_loader, args, groundTruth=GT, visulation=True)
elif args.backbone in args.SMISO:
    tester.linear_visualation_SMISO(model, data_loader, args, groundTruth=GT, visulation=True)
elif args.backbone in args.MMISO:
    tester.linear_visualation_MMISO(model, data_loader, args, groundTruth=GT, visulation=True)
elif args.backbone in args.MMIMO:
    tester.linear_visualation_MMIMO(model, data_loader, args, groundTruth=GT, visulation=True)














