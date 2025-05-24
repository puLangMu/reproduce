import os
import argparse
from importlib import import_module
import cv2
import sys
import matplotlib.pyplot as plt  # 导入 matplotlib
import torch.nn.functional as F  # 用于插值

from dataset import *

Benchmark = "organizedData"
ImageSize = (512,512)
BatchSize = 2
NJobs = 8

train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)

# 从 train_loader 中抽取一个批次
for batch in val_loader:
    mask, source, resist = batch  # 获取一个批次的数据
    break  # 只需要一个批次，退出循环

# 将一个批次中的图片显示为对照图
fig, axes = plt.subplots(BatchSize, 3, figsize=(12, 8))  # 创建子图 (BatchSize行3列)
for i in range(BatchSize):
    # 提取每组图片
    single_mask = mask[i].squeeze().cpu().numpy()  # 去掉通道维度
    single_source = source[i].unsqueeze(0)  # 添加批次维度以便插值
    single_resist = resist[i].squeeze().cpu().numpy()

    print(single_source.shape)
    # 将 source 图像调整为 (256, 256)
    single_source_resized = F.interpolate(
        single_source,  # 添加批次维度
        size=(256, 256), 
        mode="bilinear", 
        align_corners=False
    )
    single_source_resized = single_source_resized.squeeze(0).squeeze(0).cpu().numpy()  # 去掉批次和通道维度

    # 显示 mask 图像
    axes[i, 0].imshow(single_mask, cmap="gray")
    axes[i, 0].set_title(f"Mask {i+1}")
    axes[i, 0].axis("off")

    # 显示调整后的 source 图像
    axes[i, 1].imshow(single_source_resized, cmap="gray")
    axes[i, 1].set_title(f"Source (256x256) {i+1}")
    axes[i, 1].axis("off")

    # 显示 resist 图像
    axes[i, 2].imshow(single_resist, cmap="gray")
    axes[i, 2].set_title(f"Resist {i+1}")
    axes[i, 2].axis("off")

# 调整布局并显示
plt.tight_layout()
plt.show()
