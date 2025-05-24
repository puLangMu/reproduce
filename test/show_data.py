import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 设置要读取的图片索引
i =  6 # 可以根据需求修改这个变量

# organizedData文件夹路径
base_dir = "./organizedData"

# 获取四个子文件夹
folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))][:4]

# 创建一个2行4列的图像网格，减小整体figsize
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

# 从每个文件夹读取第i张图片
for col, folder in enumerate(folders):
    folder_path = os.path.join(base_dir, folder)
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if i < len(image_files):
        img_path = os.path.join(folder_path, image_files[i])
        # 读取图片并转换为灰度图
        img = Image.open(img_path).convert('L')
        
        # 调整图像尺寸为128x128（降低尺寸）
        img_small = img.resize((256, 256), Image.LANCZOS)
        
        # 调整图像尺寸为512x512（降低尺寸）
        img_large = img.resize((1024, 1024), Image.LANCZOS)
        
        # 在第一行显示小尺寸图像(128x128)
        axes[0, col].imshow(np.array(img_small), cmap='gray')
        axes[0, col].set_title(f"{folder} (256, 256)")
        axes[0, col].axis('off')
        
        # 在第二行显示大尺寸图像(512x512)
        axes[1, col].imshow(np.array(img_large), cmap='gray')
        axes[1, col].set_title(f"{folder} (1024, 1024)")
        axes[1, col].axis('off')
    else:
        # 如果文件夹中没有足够的图片，则留空
        axes[0, col].axis('off')
        axes[1, col].axis('off')
        axes[0, col].set_title(f"{folder} - 没有第{i}张图片")

plt.tight_layout()
plt.show()
