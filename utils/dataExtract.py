import os
import shutil

# 定义路径
base_dir = "/home/plum/reproduce/LMLithoData"
output_dir = "/home/plum/reproduce/organizedData"

# 确保输出目录存在
subfolders = ["Layout", "Mask", "Source", "Resist"]
for subfolder in subfolders:
    os.makedirs(os.path.join(output_dir, subfolder), exist_ok=True)

# 遍历 LMLithoData 目录下的文件夹
folder_index = 1
for folder in sorted(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        # 获取文件夹中的图片
        images = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        if len(images) >= 4:
            # 分别提取前四张图片 layout mask resist source 
            resist_img = images[0]
            layout_img = images[1]
            mask_img  = images[2]
            source_img = images[3]

            # 复制到目标目录并重命名
            shutil.copy(os.path.join(folder_path, layout_img), os.path.join(output_dir, "Layout", f"{folder_index}.png"))
            shutil.copy(os.path.join(folder_path, mask_img), os.path.join(output_dir, "Mask", f"{folder_index}.png"))
            shutil.copy(os.path.join(folder_path, source_img), os.path.join(output_dir, "Source", f"{folder_index}.png"))
            shutil.copy(os.path.join(folder_path, resist_img), os.path.join(output_dir, "Resist", f"{folder_index}.png"))

            folder_index += 1

print("图片提取和组织完成！")