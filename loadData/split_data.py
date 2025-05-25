import torch
import random
import numpy as np


def sample_gt(gt, train_num=50, train_ratio=0.1, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    # print("test_gt", test_gt.shape)

    if mode == 'number':
        print("split_type: ", mode, "train_number: ", train_num)
        sample_num = train_num
        for c in np.unique(gt):
            if c == 0:
              continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices)) 
            y = gt[indices].ravel()  
            np.random.shuffle(X)

            max_index = np.max(len(y)) + 1
            if sample_num > max_index:
                sample_num = 15
            else:
                sample_num = train_num

            train_indices = X[: sample_num]
            test_indices = X[sample_num:]

            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]

            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    # elif mode == 'ratio':
    #     print("split_type: ", mode, "train_ratio: ", train_ratio)
    #     for c in np.unique(gt):
    #         if c == 0:
    #           continue
    #         indices = np.nonzero(gt == c)
    #         X = list(zip(*indices)) 
    #         y = gt[indices].ravel()   
    #         np.random.shuffle(X)

    #         train_num = np.ceil(train_ratio * len(y)).astype('int')
    #         # print(train_num)

    #         train_indices = X[: train_num]
    #         test_indices = X[train_num:]
            
    #         train_indices = [list(t) for t in zip(*train_indices)]
    #         test_indices = [list(t) for t in zip(*test_indices)]
    #         # print("test_indices", test_indices)

    #         train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    #         test_gt[tuple(test_indices)] = gt[tuple(test_indices)]


    elif mode == 'ratio':
            unique_classes = np.unique(gt)
            unique_classes = unique_classes[unique_classes != 0]  # skip background (0)

            train_coords = []
            test_coords = []

            for c in unique_classes:
                class_coords = list(zip(*np.where(gt == c)))
                n_total = len(class_coords)
                n_train = int(np.round(train_ratio * n_total))
                random.seed(3407)
                random.shuffle(class_coords)
                train_coords.extend(class_coords[:n_train])
                test_coords.extend(class_coords[n_train:])

            train_indices = [list(t) for t in zip(*train_coords)]
            test_indices = [list(t) for t in zip(*test_coords)]

            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

            
    # elif mode == 'disjoint':
    #     print("split_type: ", mode, "train_ratio: ", train_ratio)
    #     train_gt = np.copy(gt)
    #     test_gt = np.copy(gt)
    #     for c in np.unique(gt):
    #         mask = gt == c
    #         for x in range(gt.shape[0]):
    #             # numpy.count_nonzero 是用于统计数组中非零元素的个数
    #             first_half_count = np.count_nonzero(mask[:x, :])
    #             second_half_count = np.count_nonzero(mask[x:, :])
    #             try:
    #                 ratio = first_half_count / (first_half_count + second_half_count)
    #                 if ratio >= train_ratio:
    #                     break
    #             except ZeroDivisionError:
    #                 continue
    #         mask[:x, :] = 0
    #         train_gt[mask] = 0
    #     test_gt[train_gt > 0] = 0

    elif mode == 'disjoint':
        print("split_type: ", mode, "\ntrain_ratio: ", train_ratio)
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        
        for c in np.unique(gt):
            if c == 0:
                continue  # 忽略背景类
            mask = gt == c
            total = np.count_nonzero(mask)
            
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = total - first_half_count
                try:
                    ratio = first_half_count / total
                    if ratio >= train_ratio:
                        break
                except ZeroDivisionError:
                    continue
            
            # 如果划分后测试集没有样本，调整分割点以保留至少一个测试样本
            if second_half_count == 0:
                x = max(1, x - 1)  # 回退一行以保证测试集不为空
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = total - first_half_count

            # 再检查：如果train或test都为空，就跳过这个类
            if first_half_count == 0 or second_half_count == 0:
                print(f"[Warning] Class {c} cannot be split properly. Skipping.")
                train_gt[mask] = 0
                test_gt[mask] = 0
                continue

            # 应用分割：保留上半部分为训练，其余为测试
            mask[:x, :] = 0  # 下半部分保留
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0  # 删除测试集中与训练集重复的部分

    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))

    return train_gt, test_gt






















































