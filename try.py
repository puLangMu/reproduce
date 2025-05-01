import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import *

from utils import calculate_loss


 
from segment_anything.build_Litho import build_litho

def validate_binary_tensor(tensor):
    unique_values = torch.unique(tensor)
    print("Unique values in tensor:", unique_values)
    if torch.all((unique_values == 0) | (unique_values == 1)):
        print("The tensor is correctly binarized (only contains 0 and 1).")
    else:
        print("The tensor contains values other than 0 and 1!")

Benchmark = "organizedData"
ImageSize = (1024,1024)
BatchSize = 4
NJobs = 8

train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)

import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt

def show_images(pred, single_mask, single_source, single_resist, save_dir="pictures"):
    # 将张量从 (1, 1, 1024, 1024) 转换为 (1024, 1024) 的 numpy 数组
    pred_np = pred.squeeze().cpu().numpy()
    single_mask_np = single_mask.squeeze().cpu().numpy()
    single_source_np = single_source.squeeze().cpu().numpy()
    single_resist_np = single_resist.squeeze().cpu().numpy()

   

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建一个 1x4 的子图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 显示每张图片
    axes[0].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Prediction")
    axes[0].axis("off")

    axes[1].imshow(single_mask_np, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(single_source_np, cmap='binary')
    axes[2].set_title("Source")
    axes[2].axis("off")

    axes[3].imshow(single_resist_np, cmap='binary')
    axes[3].set_title("Resist")
    axes[3].axis("off")

    # 保存图像到文件
    save_path = os.path.join(save_dir, "comparison.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Image saved to {save_path}")
    plt.close()


x = torch.randn(1, 1, 1024, 1024)
x = torch.sigmoid(1000000*x)
y = torch.randn(1, 1, 1024, 1024)
y = torch.sigmoid(y)
y = torch.where(y > 0.5, torch.ones_like(y), torch.zeros_like(y))
z = torch.randn(1, 1, 1024, 1024)
z = torch.sigmoid(z)
z = torch.where(z > 0.5, torch.tensor(1), torch.tensor(0))

a = torch.randint(0, 2, (1, 1, 1024, 1024))
show_images(x, y, z, a)
