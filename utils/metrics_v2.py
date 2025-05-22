# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.eps = 1e-8
        # self.eps = 0

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask].astype(int),
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        # print("hist", hist)

        # 逐类别的 precision
        acc_cls = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + self.eps) # (TP,TN)/(TP+FN,FP+TN)
        # print("acc_cls", acc_cls)


        # 宏观平均更适合于平衡的数据集
        # mAcc_cls = np.nanmean(acc_cls)
        # print("mAcc_cls", mAcc_cls)

        # freq = hist.sum(axis=1) / hist.sum()
        # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        
        # 这个 IOU 和下面的 IOU 是一样的, 只是计算了类别的 IOU
        # iou = np.diag(self.confusion_matrix) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # # print("iu", iu)
        # mean_iou = np.nanmean(iou)
        # print("mean_iou", mean_iou)
        # cls_iou = dict(zip(range(self.n_classes), iou))
        # print("cls_iou", cls_iou)


        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)  # (TP+TN)/(TP+FP+TN+FN)
        # OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)

        # 这是一种针对特定的 2x2 混淆矩阵的计算方法，不适用于多分类问题，计算微观指标（适用于样本不均衡问题）
        # TP, FP, FN, TN = hist[1][1], hist[0][1], hist[1][0], hist[0][0]

        # 这是一种针对通用混淆矩阵的计算方法，计算宏观指标（适用于样本均衡问题）（本来是上面的，以后还是海泽下面的更通用版本吧）
        TP = np.diag(self.confusion_matrix)
        FP = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        FN = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        TN = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        # print("tp, fp, tn, fn", TP, FP, FN, TN)
        
        # 方法二得到的是总体的 Precision 值（宏观）
        Precision = TP / (TP + FP + self.eps)
        Recall = TP / (TP + FN + self.eps)
        F1 = 2 * Precision * Recall / (Precision + Recall + self.eps)

        po = (TP + TN) / (TP + TN + FP + FN)
        pe = ((TP + FP) * (TP + FN) + (FP + TN) * (FN + TN)) / (TP + TN + FP + FN) ** 2
        Kappa = (po - pe) / (1 - pe)

        IoU = TP / (TP + FN + FP + self.eps)
        mIoU = 0.5 * (IoU + (TN / (TN + FP + FN)))
        # print("mIoU", mIoU)
        # PA = (TP+TN)/(TP+TN+FP+FN)
        # FWIoU = ((TP+FN)/(TP+TN+FP+FN)*IoU)+((TN+FP)/(TP+TN+FP+FN)*(TN/(TN+FP+FN)))
        # MPA = 0.5*(TP/(TP+FN)+TN/(FP+TN))

        return (
            {
                # "Mean Acc/MPA : \t": acc_cls,
                "classAcc 0 :": acc_cls[0],
                "classAcc 1 :": acc_cls[1],
                # "MPA : \t": MPA,
                # "FreqW Acc/FWIoU : \t": fwavacc,
                # "FWIoU : \t": FWIoU,
                # "Mean IoU/mIoU : \t": mean_iu,
                "OA: \t\t": OA,
                "Precision : ": Precision,
                "Recall : \t": Recall,
                "F1: \t\t": F1,
                "Kappa: \t\t": Kappa, 
                # "IoU : \t\t": IoU,
                "mIoU : \t\t": mIoU,
            },
            acc_cls,
        )
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':

    # gt = np.array([[0, 2, 1],
    #                [1, 2, 1],
    #                [1, 0, 1]])

    # pre = np.array([[0, 1, 1],
    #                [2, 0, 1],
    #                [1, 1, 1]])
    
    # gt = np.array([0, 0, 1, 1, 1, 1, 1, 0])
    # pre = np.array([0, 1, 0, 1, 1, 1, 1, 0])
    gt = np.array([0, 0, 1, 1, 1, 1])
    pre = np.array([0, 1, 0, 1, 1, 1])
    # print(gt.shape, pre.shape)

    running_metrics_test = runningScore(n_classes=2)
    # eval.add_batch(gt, pre)
    running_metrics_test.update(gt, pre)
    score, class_iou = running_metrics_test.get_scores()
    # print(score.items(), class_iou)

    print("classAcc 0", np.nanmean(score["classAcc 0 :"]).round(4)*100)
    print("classAcc 1", np.nanmean(score["classAcc 1 :"]).round(4)*100)
    print("OA       ", np.nanmean(score["OA: \t\t"]).round(4)*100)
    print("Precision", np.nanmean(score["Precision : "]).round(4)*100)
    print("Recall   ", np.nanmean(score["Recall : \t"]).round(4)*100)
    print("F1       ", np.nanmean(score["F1: \t\t"]).round(4)*100)
    print("Kappa    ", np.nanmean(score["Kappa: \t\t"]).round(4)*100)
    print("mIoU      ", np.nanmean(score["mIoU : \t\t"]).round(4)*100)
    print("Class IoU", class_iou)