import os
import argparse
from importlib import import_module
import cv2
import torch

import sys
import os


from segment_anything.modeling.source_encoder import SourceEncoder

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 准备测试张量 (B, C, H, W)
B, C, H, W = 1, 1, 256,256  # 批次大小为 4，单通道，图像大小为 256x256
test_input = torch.randn(B, C, H, W, device=device)  # 随机生成测试张量并移动到 GPU

# 2. 实例化 SourceEncoder 模型并移动到 GPU
model = SourceEncoder(
    img_size=256,  # 输入图像大小
    in_chans=1,    # 输入通道数
    embed_dim=64,  # 嵌入维度
    depth=2,       # 模型深度
    num_heads=8,   # 注意力头数
).to(device)  # 将模型移动到 GPU



# 3. 调用 forward 函数
output = model(test_input)

# 4. 打印输出形状
print(f"输入形状: {test_input.shape}")
print(f"输出形状: {output.shape}")

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # 打印模型的参数量
# print(f"模型参数量: {count_parameters(model):,} 个")

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.numel()} 个参数")

