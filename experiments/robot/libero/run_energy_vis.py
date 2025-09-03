import torch

# 载入
pixel_values = torch.load("energy_vis/pixel_values.pt")
print(pixel_values.shape)