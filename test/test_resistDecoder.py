import torch
from segment_anything.modeling.resist_decoder import ResistDecoder
from segment_anything.modeling.common import LayerNorm2d

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 准备测试张量
# 图像嵌入 (B, C, H, W)
B, C, H, W = 1, 256, 64, 64
image_embeddings = torch.randn(B, C, H, W, device=device).to(device)
source_embeddings = torch.randn(B, C, H, W, device=device).to(device)

# 稀疏提示嵌入 (num_prompts, num_tokens, transformer_dim)
num_batch, num_tokens, transformer_dim = 1, 64, 256


# 位置编码 (C, H, W)
# image_pe = torch.randn(C, H, W, device=device)

# 2. 实例化 ResistDecoder 模型

model = ResistDecoder(
    transformer_dim= transformer_dim
).to(device)

# 3. 调用 forward 函数
output = model(
    image_embeddings=image_embeddings,
    # image_pe=image_pe,
    source_embeddings=source_embeddings,
)

# 4. 打印输入和输出形状
print(f"输入形状 (image_embeddings): {image_embeddings.shape}")
# print(f"输入形状 (image_pe): {image_pe.shape}")
print(f"输入形状 (sparse_prompt_embeddings): {source_embeddings.shape}")
print(f"输出形状: {output.shape}")

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # 打印模型的参数量
# print(f"模型参数量: {count_parameters(model):,} 个")

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.numel()} 个参数")