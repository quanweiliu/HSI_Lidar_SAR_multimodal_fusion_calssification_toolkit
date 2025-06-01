import torch
import numpy as np 
import random
import spectral as spl
import matplotlib.pyplot as plt
from loadData import data_reader
from loadData.split_data import sample_gt


# only active in this file
def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 


def get_data(args):

    if args.dataset_name in args.SD:
        data, GT = data_reader.load_data(args.dataset_name, path_data=args.path_data, type_data=None)
        # print("data", data.shape, "data_gt", data_gt.shape)


        if args.backbone in args.MMISO or args.backbone in args.MMIMO:
            patch_size = args.patch_size * 3
        else:
            patch_size = args.patch_size
        pad_width = (patch_size // 2) + 1
        # pad_width = patch_size // 2

        # img = np.pad(data, pad_width=pad_width, mode="constant", constant_values=(0))
        # img = img[:, :, pad_width:img.shape[2]-pad_width]
        img = np.pad(data, ((pad_width, pad_width), (pad_width, pad_width), (0,0)), 'symmetric')

        if args.pca:
            print("pca is used")
            img, pca = data_reader.apply_PCA(img, num_components=args.components)
        else:
            print("pca is not used")


        data_gt = np.pad(GT, pad_width=pad_width, mode="constant", constant_values=(0))
        train_gt, test_gt = sample_gt(data_gt, train_num=args.train_num, 
                                train_ratio=args.train_ratio, mode=args.split_type)
        train_gt, val_gt = sample_gt(train_gt, train_num=args.train_num, 
                                train_ratio=0.5, mode="ratio")
        # print("train_gt", train_gt.shape, "test_gt", test_gt.shape)


        if args.show_gt:
            # data_reader.draw(data_gt, args.result_dir + "/" + args.dataset_name + "data_gt", save_img=True)
            plt.figure(figsize=(12, 8))
            # spl.imshow(classes=GT)
            spl.imshow(classes=train_gt)
            # spl.imshow(classes=test_gt)
            # plt.imshow(out)
            plt.axis('off')  # 关闭坐标轴（等效于关闭刻度和边框）
            plt.tight_layout(pad=0)  # 去除额外空白边距
            plt.show()

        if args.print_data_info:
            print("print_data_info : ---->")
            data_reader.data_info(train_gt, val_gt, test_gt, start=args.data_info_start)

        # return img, train_gt, val_gt, test_gt, data_gt, GT
        return img, img, train_gt, val_gt, test_gt, data_gt, GT


    elif args.dataset_name in args.MD:
        data1, data2, GT = data_reader.load_data(args.dataset_name, path_data=args.path_data, type_data="GT")
        data1, data2, train_gt = data_reader.load_data(args.dataset_name, path_data=args.path_data, type_data="TRLabel")
        data1, data2, test_gt = data_reader.load_data(args.dataset_name, path_data=args.path_data, type_data="TSLabel")
        # print("data", data.shape, "data_gt", data_gt.shape)

        if args.backbone in args.MMISO or args.backbone in args.MMIMO:
            patch_size = args.patch_size * 3
        else:
            patch_size = args.patch_size
        pad_width = (patch_size // 2) + 1
        # pad_width = patch_size // 2

        img1 = np.pad(data1, ((pad_width, pad_width), (pad_width, pad_width), (0,0)), 'symmetric')
        img2 = np.pad(data2, ((pad_width, pad_width), (pad_width, pad_width), (0,0)), 'symmetric')
        # img1 = np.pad(data1, pad_width=pad_width, mode="constant", constant_values=(0))
        # img2 = np.pad(data2, pad_width=pad_width, mode="constant", constant_values=(0))
        # img1 = img1[:, :, pad_width:img1.shape[2]-pad_width]
        # img2 = img2[:, :, pad_width:img2.shape[2]-pad_width]
        # print(img1.shape, img2.shape)

        if args.pca:
            print("pca is used")
            img1, pca = data_reader.apply_PCA(img1, num_components=args.components)
        else:
            print("pca is not used")

        data_gt = np.pad(GT, pad_width=pad_width, mode="constant", constant_values=(0))
        train_gt = np.pad(train_gt, pad_width=pad_width, mode="constant", constant_values=(0))
        test_gt = np.pad(test_gt, pad_width=pad_width, mode="constant", constant_values=(0))

        # data_gt, _ = sample_gt(data_gt, train_num=args.train_num, 
        #                         train_ratio=args.train_ratio, mode=args.split_type)
        train_gt, val_gt = sample_gt(train_gt, train_num=args.train_num, 
                                train_ratio=args.train_ratio, mode=args.split_type)
        test_gt, _ = sample_gt(test_gt, train_num=args.train_num, 
                                train_ratio=1, mode="ratio")   
        # print("train_gt", train_gt.shape, "test_gt", test_gt.shape)


        if args.show_gt:
            # data_reader.draw(data_gt, args.result_dir + "/" + args.dataset_name + "data_gt", save_img=True)
            plt.figure(figsize=(12, 8))
            # spl.imshow(classes=GT)
            spl.imshow(classes=train_gt)
            # spl.imshow(classes=test_gt)
            # plt.imshow()
            plt.axis('off')  # 关闭坐标轴（等效于关闭刻度和边框）
            plt.tight_layout(pad=0)  # 去除额外空白边距
            plt.show()


        if args.print_data_info:
            print("print_data_info : ---->")
            data_reader.data_info(train_gt, val_gt, test_gt, start=args.data_info_start)

        return img1, img2, train_gt, val_gt, test_gt, data_gt, GT

# 单模态，多模态，单尺度
class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data1, gt, transform, patch_size=5, data2=None, remove_zero_labels=True):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
            mixture_augmentation  不能用
        """
        super(HyperX, self).__init__()
        self.data1 = data1
        self.data2 = data2
        self.label = gt
        self.transform = transform
        self.patch_size = patch_size
        self.ignored_labels = set()
        self.center_pixel = True
        self.remove_zero_labels = remove_zero_labels
    
        mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2

        self.indices = np.array(
            [
                (x, y) for x, y in zip(x_pos, y_pos)
                if x > p and x < data1.shape[0] - p - 1 and y > p and y < data1.shape[1] - p - 1
                # if x >= p and x < data1.shape[0] - p and y >= p and y < data1.shape[1] - p
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]

        # remove zero labels, 这里删除是通过 self.indices 删除的，不是通过 self.labels 删除的
        if self.remove_zero_labels:
            self.indices = np.array(self.indices)
            self.labels = np.array(self.labels)

            self.indices = self.indices[self.labels>0]
            self.labels = self.labels[self.labels>0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        '''
            x, y -> index
            x1, y1 = x - 4, y - 4
            x2, y2 = x, y
        '''
        x, y = self.indices[index]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data1 = self.data1[x1:x2, y1:y2]
        if isinstance(self.data2, np.ndarray):
            data2 = self.data2[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data1 = np.copy(data1).transpose((2, 0, 1))
        if isinstance(self.data2, np.ndarray):
            data2 = np.copy(data2).transpose((2, 0, 1))
        label = label

        # Load the data into PyTorch tensors
        data1 = torch.from_numpy(data1)
        if isinstance(self.data2, np.ndarray):
            data2 = torch.from_numpy(data2)
        label = torch.from_numpy(label)

        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data1 = data1[:, 0, 0]
            if isinstance(self.data2, np.ndarray):
                data2 = data2[:, 0, 0]
            label = label[0, 0]
        
        
        if self.transform != None:
            # print("transformed", )
            data1 = self.transform(data1)
            if isinstance(self.data2, np.ndarray):
                data2 = self.transform(data2)

        if isinstance(self.data2, np.ndarray):
            return data1, data2, label
        else:
            return data1, label
    

# 单模态，多模态，多尺度
class HyperXMM(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data1, gt, transform, patch_size=5, data2=None, remove_zero_labels=True):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
            mixture_augmentation  不能用
        """
        super(HyperXMM, self).__init__()
        self.data1 = data1
        self.data2 = data2
        self.label = gt
        self.transform = transform
        self.patch_size = patch_size
        self.patch_sizeX2 = patch_size * 2
        self.patch_sizeX3 = patch_size * 3
        self.ignored_labels = set()
        self.center_pixel = True
        self.remove_zero_labels = remove_zero_labels
    
        mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_sizeX3 // 2

        self.indices = np.array(
            [
                (x, y) for x, y in zip(x_pos, y_pos)
                if x > p and x < data1.shape[0] - p - 1 and y > p and y < data1.shape[1] - p - 1
                # if x >= p and x < data1.shape[0] - p and y >= p and y < data1.shape[1] - p
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]

        # remove zero labels, 这里删除是通过 self.indices 删除的，不是通过 self.labels 删除的
        if self.remove_zero_labels:
            self.indices = np.array(self.indices)
            self.labels = np.array(self.labels)

            self.indices = self.indices[self.labels>0]
            self.labels = self.labels[self.labels>0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        '''
            x, y -> index
            x1, y1 = x - 4, y - 4
            x2, y2 = x, y
        '''
        x, y = self.indices[index]
        # print("self.patch_size", self.patch_size)
        x11, y11 = x - self.patch_size // 2, y - self.patch_size // 2
        x12, y12 = x - self.patch_sizeX2 // 2, y - self.patch_sizeX2 // 2
        x13, y13 = x - self.patch_sizeX3 // 2, y - self.patch_sizeX3 // 2
        x21, y21 = x11 + self.patch_size, y11 + self.patch_size
        x22, y22 = x12 + self.patch_sizeX2, y12 + self.patch_sizeX2
        x23, y23 = x13 + self.patch_sizeX3, y13 + self.patch_sizeX3
        # print("self.patch_size", x11, x21, y11, y21)

        data11 = self.data1[x11:x21, y11:y21].transpose((2, 0, 1))
        data12 = self.data1[x12:x22, y12:y22].transpose((2, 0, 1))
        data13 = self.data1[x13:x23, y13:y23].transpose((2, 0, 1))
        if isinstance(self.data2, np.ndarray):
            data21 = self.data2[x11:x21, y11:y21].transpose((2, 0, 1))
            data22 = self.data2[x12:x22, y12:y22].transpose((2, 0, 1))
            data23 = self.data2[x13:x23, y13:y23].transpose((2, 0, 1))
        label = self.label[x11:x21, y11:y21]
        # label2 = self.label[x12:x22, y12:y22]
        # label3 = self.label[x13:x23, y13:y23]

        # # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        # data1 = data1.transpose((2, 0, 1))
        # if isinstance(self.data2, np.ndarray):
        #     data2 = data2.transpose((2, 0, 1))
        # label = label

        # Load the data into PyTorch tensors
        data11 = torch.from_numpy(data11)
        data12 = torch.from_numpy(data12)
        data13 = torch.from_numpy(data13)
        if isinstance(self.data2, np.ndarray):
            data21 = torch.from_numpy(data21)
            data22 = torch.from_numpy(data22)
            data23 = torch.from_numpy(data23)
        label = torch.from_numpy(label)
        # label2 = torch.from_numpy(label2)
        # label3 = torch.from_numpy(label3)

        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
            # label2 = label2[self.patch_sizeX2 // 2, self.patch_sizeX2 // 2]
            # label3 = label3[self.patch_sizeX3 // 2, self.patch_sizeX3 // 2]
        
        # # Remove unused dimensions when we work with invidual spectrums
        # elif self.patch_size == 1:
        #     data1 = data1[:, 0, 0]
        #     if isinstance(self.data2, np.ndarray):
        #         data2 = data2[:, 0, 0]
        #     label = label[0, 0]
        
        if self.transform != None:
            # print("transformed", )
            data1 = self.transform(data1)
            if isinstance(self.data2, np.ndarray):
                data2 = self.transform(data2)

        if isinstance(self.data2, np.ndarray):
            return data11, data12, data13, data21, data22, data23, label
        else:
            return data11, data12, data13, label







