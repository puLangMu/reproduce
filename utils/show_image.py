import matplotlib.pyplot as plt
import os

def show_images(pred, single_mask, single_source, single_resist, save_dir="pictures", name = "comparison"):
    # 将张量从 (1, 1, 1024, 1024) 转换为 (1024, 1024) 的 numpy 数组
    pred_np = pred.squeeze().cpu().numpy()
    single_mask_np = single_mask.squeeze().cpu().numpy()
    single_source_np = single_source.squeeze().cpu().numpy()
    single_resist_np = single_resist.squeeze().cpu().numpy()

   

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建一个 1x4 的子图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 显示每张图片
    axes[0].imshow(pred_np, cmap='gray')
    axes[0].set_title("Prediction")
    axes[0].axis("off")

    axes[1].imshow(single_mask_np, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(single_source_np, cmap='gray')
    axes[2].set_title("Source")
    axes[2].axis("off")

    axes[3].imshow(single_resist_np, cmap='gray')
    axes[3].set_title("Resist")
    axes[3].axis("off")

    # 保存图像到文件
    save_path = os.path.join(save_dir, f"{name}.png")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Image saved to {save_path}")
    plt.close()