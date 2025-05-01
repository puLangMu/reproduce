import torch
import sys


x = torch.randn(1, 3, 224, 224, device='cuda',requires_grad=True)

y = torch.randn(1, 3, 224, 224, device='cuda',requires_grad=True)

z = torch.where(x > 0, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))


z.backward(torch.ones_like(z))

print(x.grad)