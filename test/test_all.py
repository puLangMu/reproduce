import torch
import sys

sys.path.append("..")
from segment_anything import build_litho

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 准备输入张量
# 图像输入 (B, C, H, W)
B, C, H, W = 1, 1, 1024, 1024
image_input = torch.randn(B, C, H, W, device=device, requires_grad=True)

# Source 输入 (B, C, H, W)
source_input = torch.randn(B, C, 256, 256, device=device, requires_grad=True)

# 稀疏提示嵌入 (num_prompts, num_tokens, transformer_dim)
num_prompts, num_tokens, transformer_dim = 1, 128, 256



model = build_litho()
model.train()

target = torch.clamp(torch.randn(B, 1, 1024, 1024, device=device, requires_grad=True), 0, 1)
loss_fn = torch.nn.BCELoss()



for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"Parameter {name} does not require gradient.")
output = model(image_input, source_input)


loss = loss_fn(output, target)
loss.backward(retain_graph=True)
print(f"Loss: {loss.item()}")

print(f"Model 输出形状: {output.shape}")
print(output)


for name, parms in reversed(list(model.named_parameters())):
    print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
        

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # 打印模型的参数量
# print(f"模型参数量: {count_parameters(model):,} 个")

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.numel()} 个参数")
