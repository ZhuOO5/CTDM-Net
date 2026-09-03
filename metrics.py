from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
import torchvision.transforms as transforms
import os


def update_seg_metrics(total_inter,total_union,total_correct,total_label,
                       total_FPR,total_pred,total_lab_class,
                       correct,labeled,inter,union,FPR,pred,lab_class):

    total_correct += correct
    total_label += labeled
    total_inter += inter
    total_union += union
    total_FPR += FPR
    total_pred += pred
    total_lab_class += lab_class   # ⭐ 新增：累计每类GT像素

    return total_inter,total_union,total_correct,total_label,total_FPR,total_pred,total_lab_class


def get_seg_metrics(total_correct,total_label,
                    total_inter,total_union,
                    total_FPR,total_pred,
                    total_lab_class,
                    num_classes):

    eps = np.spacing(1)

    pixAcc = 1.0 * total_correct / (eps + total_label)
    IoU = 1.0 * total_inter / (eps + total_union)
    FPR = 1.0 * total_FPR / (eps + total_pred)
    F1 = 2 * IoU / (IoU + 1)

    mF1 = F1.mean()
    mFPR = FPR.mean()
    mIoU = IoU.mean()

    # ===== Fscd =====
    inter_scd = np.sum(total_inter[1:])        # TP
    pred_scd  = np.sum(total_pred[1:])         # TP + FP
    label_scd = np.sum(total_lab_class[1:])    # TP + FN

    P_scd = inter_scd / (pred_scd + eps)
    R_scd = inter_scd / (label_scd + eps)
    Fscd  = (2 * P_scd * R_scd) / (P_scd + R_scd + eps)

    return {
        "Pixel_Accuracy": np.round(pixAcc, 5),
        "Mean_FPR":np.round(mFPR,5),
        "Mean_IoU": np.round(mIoU, 5),
        "Mean_F1":np.round(mF1,5),
        "Fscd": np.round(Fscd,5),
        "Class_IoU": dict(zip(range(num_classes), np.round(IoU, 5))),
        "Class_F1":dict(zip(range(num_classes), np.round(F1, 5)))
    }


def batch_pix_accuracy(predict, target, labeled):

    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()
    assert pixel_correct <= pixel_labeled

    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def batch_intersection_union(predict, target, num_class, labeled):

    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(),
                             bins=num_class,
                             max=num_class,
                             min=1)

    area_pred = torch.histc(predict.float(),
                            bins=num_class,
                            max=num_class,
                            min=1)

    area_lab = torch.histc(target.float(),
                           bins=num_class,
                           max=num_class,
                           min=1)

    area_union = area_pred + area_lab - area_inter
    area_FPR = area_pred - area_inter

    assert (area_inter <= area_union).all()

    return (area_inter.cpu().numpy(),
            area_union.cpu().numpy(),
            area_FPR.cpu().numpy(),
            area_pred.cpu().numpy(),
            area_lab.cpu().numpy())   # ⭐ 新增返回


def eval_metrics(output, target, num_class):

    _, predict = torch.max(output.data, 1)

    predict = predict + 1
    target = target + 1

    labeled = (target > 0) * (target <= num_class)

    correct, num_labeled = batch_pix_accuracy(predict, target, labeled)

    inter, union, area_FPR, area_pred, area_lab = \
        batch_intersection_union(predict, target, num_class, labeled)

    return [np.round(correct, 5),
            np.round(num_labeled, 5),
            np.round(inter, 5),
            np.round(union, 5),
            np.round(area_FPR, 5),
            np.round(area_pred, 5),
            np.round(area_lab, 5)]   # ⭐ 多返回 lab_class


def eval_metrics_direct(predict, target, num_class):

    predict = predict + 1
    target = target + 1

    labeled = (target > 0) * (target <= num_class)

    correct, num_labeled = batch_pix_accuracy(predict, target, labeled)

    inter, union, area_FPR, area_pred, area_lab = \
        batch_intersection_union(predict, target, num_class, labeled)

    return [np.round(correct, 5),
            np.round(num_labeled, 5),
            np.round(inter, 5),
            np.round(union, 5),
            np.round(area_FPR, 5),
            np.round(area_pred, 5),
            np.round(area_lab, 5)]
