import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import *

from utils import calculate_loss


 
from segment_anything.build_Litho import build_litho, build_source_litho

def validate_binary_tensor(tensor):
    unique_values = torch.unique(tensor)
    print("Unique values in tensor:", unique_values)
    if torch.all((unique_values == 0) | (unique_values == 1)):
        print("The tensor is correctly binarized (only contains 0 and 1).")
    else:
        print("The tensor contains values other than 0 and 1!")

Benchmark = "organizedData"
ImageSize = (1024,1024)
BatchSize = 7
NJobs = 8

train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)

import matplotlib.pyplot as plt

import os
from utils import show_images




# 调用函数展示图片

def main():

    for step, batch in enumerate(val_loader):
        mask, source, resist = batch  # 获取一个批次的数据
        # 假设批次大小为 4，提取第一张图片
        # single_mask = mask[1:2,:,:,:]  # 第一张 mask 图像
        # single_source = source[1:2,:,:,:]  # 第一张 source 图像
        # single_resist = resist[1:2,:,:,:] # 第一张 resist 图像
        if step > 2:
            break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    source = torch.nn.functional.interpolate(source, size=(256, 256), mode='bilinear', align_corners=False)

    

    k = 0.7
    alpha = 2
    beta = 1
    gamma = 3


    # create model
    model = build_litho().to(device)
    # load model weights
    model_weight_path = "./saved/direct_k07.pth"
    # model_weight_path = "./saved/k09.pth"

    # model_weight_path = "./weights/model-24.pth"

    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # with torch.no_grad():
    #     # predict class



    #     pred = model(single_mask.to(device), single_source.to(device)) 
    #     # pred = torch.sigmoid((pred - 0.5) * 100)

    with torch.no_grad():
    # 存储所有预测结果的列表
        preds = []

        for i in range(mask.shape[0]):
            # 获取单张 mask 和 source
            single_mask = mask[i:i+1, :, :, :].to(device)
            single_source = source[i:i+1, :, :, :].to(device)

            # 进行预测
            pred = model(single_mask, single_source)

            # 将预测结果添加到列表中
            preds.append(pred)

        # 将预测结果列表转换为张量

        # 继续后续处理

        pred = preds[0]  # 取出第一张预测结果
        single_mask = mask[0:1, :, :, :].to(device)
        single_resist = resist[0:1, :, :, :].to(device)
        single_source = source[0:1, :, :, :].to(device)

        loss, BCE_loss, dice_loss, ssim_loss = calculate_loss(pred, single_resist.to(device), alpha = alpha, beta = beta, gamma = gamma, k= k)
        print("Loss:", loss.item())
        print("BCE Loss:", BCE_loss.item())
        print("Dice Loss:", dice_loss.item())
        print("SSIM Loss:", ssim_loss.item())

        loss, BCE_loss, dice_loss, ssim_loss = calculate_loss(single_resist.to(device), single_resist.to(device), alpha=alpha, beta=beta, gamma=gamma, k=k)
        print("Loss(Truth):", loss.item())
        print("BCE Loss(Truth):", BCE_loss.item())
        print("Dice Loss(Truth):", dice_loss.item())
        print("SSIM Loss(Truth):", ssim_loss.item())

        loss, BCE_loss, dice_loss, ssim_loss = calculate_loss(single_mask.to(device), single_resist.to(device), alpha=alpha, beta=beta, gamma=gamma, k=k)
        print("Loss(mask):", loss.item())
        print("BCE Loss(mask):", BCE_loss.item())
        print("Dice Loss(mask):", dice_loss.item())
        print("SSIM Loss(mask):", ssim_loss.item())

        loss, BCE_loss, dice_loss, ssim_loss = calculate_loss(pred, single_mask.to(device), alpha=alpha, beta=beta, gamma=gamma, k=k)
        print("Loss(another):", loss.item())
        print("BCE Loss(another):", BCE_loss.item())
        print("Dice Loss(another):", dice_loss.item())
        print("SSIM Loss(another):", ssim_loss.item())
    

   # 确保 pred 是二值化的整数类型
    # pred = torch.where(pred > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
    # pred = pred.to(torch.uint8)  # 转换为整数类型

    # 验证 pred 是否为二值化的张量
    # validate_binary_tensor(pred)

    # 保存图片到文件夹
    # show_images(pred, single_mask, single_source, single_resist, save_dir="pictures")
    
    for i in range(mask.shape[0]):
        show_images(preds[i], mask[i:i+1, :, :, :], source[i:i+1, :, :, :], resist[i:i+1, :, :, :], save_dir="pictures", name = f"pred_resist_{i}.png")
        # show_images(preds[i], mask[i:i+1, :, :, :], source[i:i+1, :, :, :], resist[i:i+1, :, :, :], save_dir="pictures", name = f"pred_{i}.png")


if __name__ == '__main__':
    main()
    

