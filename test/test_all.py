import torch
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.source_encoder import SourceEncoder
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.modeling.common import LayerNorm2d

from segment_anything.build_Litho import build_litho

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 准备输入张量
# 图像输入 (B, C, H, W)
B, C, H, W = 1, 1, 1024, 1024
image_input = torch.randn(B, C, H, W, device=device)

# Source 输入 (B, C, H, W)
source_input = torch.randn(B, C, 256, 256, device=device)

# 稀疏提示嵌入 (num_prompts, num_tokens, transformer_dim)
num_prompts, num_tokens, transformer_dim = 1, 128, 256

# # 位置编码 (C, H, W)
# # image_pe = torch.randn(256, 64, 64, device=device)

# # 2. 实例化 ImageEncoderViT 模型
# image_encoder = ImageEncoderViT(
#     img_size=1024,  # 输入图像大小
#     patch_size=16,  # Patch 大小
#     in_chans=1,     # 输入通道数
#     embed_dim=768,  # 嵌入维度
#     depth=12,       # 模型深度
#     num_heads=12,   # 注意力头数
#     mlp_ratio=4.0,  # MLP 隐藏层与嵌入维度的比例
#     out_chans=256,  # 输出通道数
# ).to(device)

# # 3. 实例化 SourceEncoder 模型
# source_encoder = SourceEncoder(
#     img_size=256,  # 输入图像大小
#     in_chans=1,    # 输入通道数
#     embed_dim=64,  # 嵌入维度
#     depth=4,       # 模型深度
#     num_heads=8,   # 注意力头数
# ).to(device)

# # 4. 实例化 MaskDecoder 模型
# transformer = TwoWayTransformer(
#     depth=2,
#     embedding_dim=transformer_dim,
#     mlp_dim=2048,
#     num_heads=8,
# ).to(device)

# mask_decoder = MaskDecoder(
#     transformer_dim=transformer_dim,
#     transformer=transformer,
#     image_embedding_size=(64, 64),
#     embed_dim=256,
# ).to(device)

# # 5. 编码阶段
# # 图像编码
# image_embeddings = image_encoder(image_input)
# print(f"Image Encoder 输出形状: {image_embeddings.shape}")

# # Source 编码
# source_embeddings = source_encoder(source_input)
# print(f"Source Encoder 输出形状: {source_embeddings.shape}")

# # 6. 解码阶段
# output = mask_decoder(
#     image_embeddings=image_embeddings,
#     sparse_prompt_embeddings=source_embeddings,
# )
# print(f"Mask Decoder 输出形状: {output.shape}")

model = build_litho()
output = model(image_input, source_input)
print(f"Model 输出形状: {output.shape}")
print(output)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # 打印模型的参数量
# print(f"模型参数量: {count_parameters(model):,} 个")

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.numel()} 个参数")
