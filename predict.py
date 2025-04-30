import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import *

from utils import calculate_loss


 
from segment_anything.build_Litho import build_litho

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
    axes[0].imshow(pred_np, cmap='gray')
    axes[0].set_title("Prediction")
    axes[0].axis("off")

    axes[1].imshow(single_mask_np, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(single_source_np, cmap='gray')
    axes[2].set_title("Source")
    axes[2].axis("off")

    axes[3].imshow(single_resist_np, cmap='gray')
    axes[3].set_title("Resist")
    axes[3].axis("off")

    # 保存图像到文件
    save_path = os.path.join(save_dir, "comparison.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Image saved to {save_path}")
    plt.close()

# 调用函数展示图片

def main():

    for batch in train_loader:
        mask, source, resist = batch  # 获取一个批次的数据
        # 假设批次大小为 4，提取第一张图片
        single_mask = mask[:1,:,:,:]  # 第一张 mask 图像
        single_source = source[:1,:,:,:]  # 第一张 source 图像
        single_resist = resist[:1,:,:,:] # 第一张 resist 图像
        break  # 只需要一个批次，退出循环

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    single_source = torch.nn.functional.interpolate(single_source, size=(256, 256), mode='bilinear', align_corners=False)

    


    # create model
    model = build_litho().to(device)
    # load model weights
    model_weight_path = "./weights/model-3.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # predict class
        pred = model(single_mask.to(device), single_source.to(device)) 


        loss, BCE_loss, dice_loss, ssim_loss = calculate_loss(pred, single_resist.to(device), 1, 1, 100, 0.7)
        print("Loss:", loss.item())
        print("BCE Loss:", BCE_loss.item())
        print("Dice Loss:", dice_loss.item())
        print("SSIM Loss:", ssim_loss.item())

        loss, BCE_loss, dice_loss, ssim_loss = calculate_loss(single_resist.to(device), single_resist.to(device), 1, 1, 100, 0.7)
        print("Loss(Truth):", loss.item())
        print("BCE Loss(Truth):", BCE_loss.item())
        print("Dice Loss(Truth):", dice_loss.item())
        print("SSIM Loss(Truth):", ssim_loss.item())

        loss, BCE_loss, dice_loss, ssim_loss = calculate_loss(single_mask.to(device), single_resist.to(device), 1, 1, 100, 0.7)
        print("Loss(mask):", loss.item())
        print("BCE Loss(mask):", BCE_loss.item())
        print("Dice Loss(mask):", dice_loss.item())
        print("SSIM Loss(mask):", ssim_loss.item())

        loss, BCE_loss, dice_loss, ssim_loss = calculate_loss(pred, single_mask.to(device), 1, 1, 100, 0.7)
        print("Loss(another):", loss.item())
        print("BCE Loss(another):", BCE_loss.item())
        print("Dice Loss(another):", dice_loss.item())
        print("SSIM Loss(another):", ssim_loss.item())
    
    pred = torch.where(pred > 0.5, torch.tensor(1.0, device=pred.device), torch.tensor(0.0, device=pred.device))
    # 保存图片到文件夹
    show_images(pred, single_mask, single_source, single_resist, save_dir="pictures")


if __name__ == '__main__':
    main()