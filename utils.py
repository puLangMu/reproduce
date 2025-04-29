import os
import sys
import json
import pickle
import random

import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from math import exp
import numpy as np
 
 
# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
 
 
# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
 
 
# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
 
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
 
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
 
    window = window.to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
 
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
 
    if full:
        return ret, cs
    return ret
 
 
 
# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
 
        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
 
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
 
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
 
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)




def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def calculate_loss(pred, labels, alpha, beta, gamma, k):
    weight = torch.full_like(pred, k, dtype=torch.float32, device=pred.device)
    
    BCE_loss_function = nn.BCELoss(weight=weight)
    BCE_loss = BCE_loss_function(pred, labels)

    dice_loss = dice_loss_function(pred, labels)

    # 计算 SSIM 损失
    ssim_temp = SSIM()
    ssim_loss = 1 - ssim_temp(pred, labels)


    loss = alpha * dice_loss + beta * BCE_loss + gamma * ssim_loss

    return loss

def dice_loss_function(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return 1-(2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    # accu_loss = torch.zeros(1).to(device)  # 累计损失
    # accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()


    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        # images, labels = data
        # sample_num += images.shape[0]

        # pred = model(images.to(device))
        # pred_classes = torch.max(pred, dim=1)[1]
        # accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # loss = loss_function(pred, labels.to(device))
        # loss.backward()
        # accu_loss += loss.detach()

        mask, source, resist = data
        source = torch.nn.functional.interpolate(source, size=(256, 256), mode='bilinear', align_corners=False)

        source.requires_grad = True
        mask.requires_grad = True
        resist.requires_grad = True
        pred = model(mask.to(device), source.to(device))

        loss = calculate_loss(pred, resist.to(device), alpha=1, beta=1, gamma=100, k=0.7)

        loss.backward()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, loss.item() )
                                                                               

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):

    model.eval()


    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
    #     images, labels = data
    #     sample_num += images.shape[0]

    #     pred = model(images.to(device))
    #     pred_classes = torch.max(pred, dim=1)[1]
    #     accu_num += torch.eq(pred_classes, labels.to(device)).sum()

    #     loss = loss_function(pred, labels.to(device))
    #     accu_loss += loss

    #     data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
    #                                                                            accu_loss.item() / (step + 1),
    #                                                                            accu_num.item() / sample_num)

    # return accu_loss.item() / (step + 1), accu_num.item() / sample_num

        mask, source, resist = data
        source = torch.nn.functional.interpolate(source, size=(256, 256), mode='bilinear', align_corners=False)

        pred = model(mask.to(device), source.to(device))

        loss = calculate_loss(pred, resist.to(device), alpha=1, beta=1, gamma=100, k=0.7)


        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, loss.item() )
                                                                               


    return loss.item()


