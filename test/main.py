import os
import argparse
from importlib import import_module
import cv2

from dataset import *

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




Benchmark = "organizedData"
ImageSize = (1024,1024)
BatchSize = 4
NJobs = 8

train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)

def dice_loss_function(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return 1-(2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def calculate_loss(pred, labels, alpha, beta, gamma, k):
    weight = torch.full_like(pred, k, dtype=torch.float32, device=pred.device)
    
    BCE_loss_function = nn.BCELoss(weight=weight)
    BCE_loss = BCE_loss_function(pred, labels)

    dice_loss = dice_loss_function(pred, labels)

    # 计算 SSIM 损失
    ssim_temp = SSIM()
    ssim_loss = 1 - ssim_temp(pred, labels)


    loss = alpha * dice_loss + beta * BCE_loss + gamma * ssim_loss

    return loss, BCE_loss, dice_loss, ssim_loss

# 从 train_loader 中抽取一张图片
for batch in train_loader:
    mask, source, resist = batch  # 获取一个批次的数据
    # 假设批次大小为 4，提取第一张图片
    single_mask = mask  # 第一张 mask 图像
    single_source = source  # 第一张 source 图像
    single_resist = resist # 第一张 resist 图像
    break  # 只需要一个批次，退出循环

loss, BCE_loss, dice_loss, ssim_loss = calculate_loss(single_mask, single_resist, 1, 1, 100, 0.7)
print("Loss:", loss.item())
print("BCE Loss:", BCE_loss.item())
print("Dice Loss:", dice_loss.item())
print("SSIM Loss:", ssim_loss.item())
# # 将一个批次中的图片显示为对照图

# single_mask_np = single_mask.squeeze().cpu().numpy()  # 去掉通道维度
# # single_source_np = single_source.squeeze().cpu().numpy()
# # single_resist_np = single_resist.squeeze().cpu().numpy()

# print(single_mask_np.shape)
# print(single_mask_np[512][512])

# # 使用 OpenCV 展示图片
# cv2.imshow("Mask", single_mask_np)
# cv2.imshow("Source", single_source_np)
# cv2.imshow("Resist", single_resist_np)

# # 等待按键关闭窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 打印形状以验证
# print("Mask shape:", single_mask.shape)
# print("Source shape:", single_source.shape)
# print("Resist shape:", single_resist.shape)

# import torch

# # 初始化变量
# pixel_sum = 0
# pixel_squared_sum = 0
# num_pixels = 0

# 遍历 train_loader 中的所有批次
# for batch in train_loader:
#     mask, source, resist = batch  # 获取一个批次的数据

#     # 将批次中的所有图片展开为一维
#     # 假设 mask、source、resist 的形状为 (BatchSize, 1, H, W)
#     mask = mask.view(mask.size(0), -1)  # 展平为 (BatchSize, H*W)
#     source = source.view(source.size(0), -1)
#     resist = resist.view(resist.size(0), -1)

#     # 计算当前批次的像素总和和平方总和
#     pixel_sum += mask.sum() + source.sum() + resist.sum()
#     pixel_squared_sum += (mask ** 2).sum() + (source ** 2).sum() + (resist ** 2).sum()

#     # 更新像素总数
#     num_pixels += mask.numel() + source.numel() + resist.numel()

# # 计算均值和方差
# mean = pixel_sum / num_pixels
# variance = (pixel_squared_sum / num_pixels) - (mean ** 2)
# std = torch.sqrt(variance)

# # 打印结果
# print(f"均值: {mean.item():.4f}")
# print(f"方差: {variance.item():.4f}")
# print(f"标准差: {std.item():.4f}")




# import sys
# sys.path.append("..")
# from segment_anything import sam_model_registry, SamPredictor
# model_type = "vit_b"
# sam = sam_model_registry[model_type]



