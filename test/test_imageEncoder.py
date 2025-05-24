import torch
from segment_anything.modeling.image_encoder import ImageEncoderViT

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 准备测试张量 (B, C, H, W)
B, C, H, W = 1, 1, 1024,1024  # 批次大小为 1，单通道，图像大小为 1024x1024
test_input = torch.randn(B, C, H, W, device=device, requires_grad= True)  # 随机生成测试张量并移动到 GPU

# 2. 实例化 ImageEncoderViT 模型并移动到 GPU
model = ImageEncoderViT(
    img_size=1024,  # 输入图像大小
    patch_size=16,  # Patch 大小
    in_chans=1,     # 输入通道数
    embed_dim=768,  # 嵌入维度
    depth=12,       # 模型深度
    num_heads=12,   # 注意力头数
    mlp_ratio=4.0,  # MLP 隐藏层与嵌入维度的比例
    out_chans=256,  # 输出通道数
    window_size = 16 # 窗口大小
).to(device)  # 将模型移动到 GPU

# # 3. 调用 forward 函数
output = model(test_input)

# # 4. 打印输入和输出形状
print(f"输入形状: {test_input.shape}")
print(f"输出形状: {output.shape}")

# 计算模型的参数量
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # 打印模型的参数量
# print(f"模型参数量: {count_parameters(model):,} 个")

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.numel()} 个参数")