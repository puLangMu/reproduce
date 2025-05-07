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


import matplotlib.pyplot as plt
import os

def show_pred_source_pairs(preds, sources, save_dir="pictures", name="pred_source_comparison"):
    """
    将 pred 和 source 对应起来，生成两行四列的图片布局。
    每一列显示 pred 和对应的 source。

    Args:
        preds (list of torch.Tensor): 预测结果的张量列表，每个张量形状为 (1, 1, H, W)。
        sources (list of torch.Tensor): 对应的 source 张量列表，每个张量形状为 (1, 1, H, W)。
        save_dir (str): 保存图片的目录。
        name (str): 保存图片的文件名（不含扩展名）。
    """
    # 确保 preds 和 sources 数量一致
    assert len(preds) == len(sources), "The number of preds and sources must be the same."

    # 将张量转换为 numpy 数组
    preds_np = [pred.squeeze().cpu().numpy() for pred in preds]
    sources_np = [source.squeeze().cpu().numpy() for source in sources]

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建一个 2xN 的子图
    num_pairs = len(preds)
    fig, axes = plt.subplots(2, num_pairs, figsize=(5 * num_pairs, 10))

    for i in range(num_pairs):
        # 显示 pred
        axes[0, i].imshow(preds_np[i], cmap='gray')
        axes[0, i].set_title(f"Prediction {i+1}")
        axes[0, i].axis("off")

        # 显示 source
        axes[1, i].imshow(sources_np[i], cmap='gray')
        axes[1, i].set_title(f"Source {i+1}")
        axes[1, i].axis("off")

    # 保存图像到文件
    save_path = os.path.join(save_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Image saved to {save_path}")
    plt.close()