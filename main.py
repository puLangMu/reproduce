import os
import argparse
from importlib import import_module
import cv2

from test.dataset import *


Benchmark = "organizedData"
ImageSize = (1024,1024)
BatchSize = 4
NJobs = 8

train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)

# 从 train_loader 中抽取一张图片
# for batch in train_loader:
#     mask, source, resist = batch  # 获取一个批次的数据
#     # 假设批次大小为 4，提取第一张图片
#     single_mask = mask[0]  # 第一张 mask 图像
#     single_source = source[0]  # 第一张 source 图像
#     single_resist = resist[0]  # 第一张 resist 图像
#     break  # 只需要一个批次，退出循环

# single_mask_np = single_mask.squeeze().cpu().numpy()  # 去掉通道维度
# single_source_np = single_source.squeeze().cpu().numpy()
# single_resist_np = single_resist.squeeze().cpu().numpy()

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




