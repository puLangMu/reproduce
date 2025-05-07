import torch

x = torch.randint(0, 10, (2, 3))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


y = (x * 2).to(device)
print(device)
print(x.device)
print(y.device)