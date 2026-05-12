import os
import glob
import math
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.fft
from torchvision import transforms



plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False



class FARM(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.freq_controller = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        fft_x = torch.fft.fft2(x, norm="ortho")
        fft_x_shifted = torch.fft.fftshift(fft_x)

        amplitude = torch.abs(fft_x_shifted)
        freq_mask = self.freq_controller(amplitude)

        fft_x_filtered = fft_x_shifted * freq_mask
        fft_x_ishifted = torch.fft.ifftshift(fft_x_filtered)
        x_restored = torch.fft.ifft2(fft_x_ishifted, norm="ortho").real

        return x + x_restored



CLASS_NAME_MAP = {
    "喉癌": "laryngocarcinoma",
    "声带白斑": "leukoplakia",
    "声带沟": "sulcus vocalis",
    "声带息肉": "polyp",
    "声带炎": "chorditis vocalis",
    "正常声带图片": "normal vocal fold",
    "声带癌": "laryngocarcinoma",
    "声带囊肿": "cyst",
    "声带乳头状瘤": "papillary carcinoma",
}



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def remove_hidden_dirs(root_dir):
    if not os.path.isdir(root_dir):
        return

    for current_root, dirnames, _ in os.walk(root_dir):
        for d in list(dirnames):
            if d.startswith(".") or "ipynb_checkpoints" in d or "__pycache__" in d:
                hidden_dir = os.path.join(current_root, d)
                shutil.rmtree(hidden_dir, ignore_errors=True)
                print(f"Removed hidden folder: {hidden_dir}")


def load_image_paths(root_dir):
    """
    支持：
    1) root_dir/*.jpg
    2) root_dir/类别/*.jpg
    并跳过隐藏目录/文件
    """
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    image_paths = []

    if not os.path.isdir(root_dir):
        return image_paths

    root_has_images = False
    for ext in exts:
        paths = glob.glob(os.path.join(root_dir, ext))
        paths = [p for p in paths if "/." not in p.replace("\\", "/")]
        if len(paths) > 0:
            root_has_images = True
            image_paths.extend(paths)

    if root_has_images:
        return sorted(image_paths)

    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
    subdirs = [
        d for d in subdirs
        if os.path.isdir(d)
        and not os.path.basename(d).startswith(".")
        and "__pycache__" not in os.path.basename(d)
        and "ipynb_checkpoints" not in os.path.basename(d)
    ]

    for sub in subdirs:
        for ext in exts:
            paths = glob.glob(os.path.join(sub, ext))
            paths = [p for p in paths if "/." not in p.replace("\\", "/")]
            image_paths.extend(paths)

    return sorted(image_paths)


def get_class_name_from_path(path, root_dir=None):
    """
    如果图片直接在 root_dir 下：
        用文件名（去后缀）作为类别名
    如果图片在子文件夹下：
        用父文件夹名作为类别名
    """
    filename = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))

    if root_dir is not None:
        root_dir = os.path.abspath(root_dir)
        parent_dir = os.path.abspath(os.path.dirname(path))
        if parent_dir == root_dir:
            return filename

    if parent.startswith(".") or "ipynb_checkpoints" in parent:
        return filename

    return parent


def map_to_english_class_name(chinese_name):
    return CLASS_NAME_MAP.get(chinese_name, chinese_name)


def tensor_to_rgb_image(tensor, mean, std):
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    return img


def diff_heatmap(img_before, img_after):
    diff = np.abs(img_after.astype(np.float32) - img_before.astype(np.float32)).mean(axis=2)
    diff = diff / (diff.max() + 1e-8)
    return diff


def metric_saturation_ratio(gray):
    return float(np.mean(gray >= 245))


def metric_tenengrad(gray):
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    fm = gx * gx + gy * gy
    return float(np.mean(fm))


def evaluate_simple_metrics(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return {
        "saturation_ratio": metric_saturation_ratio(gray),
        "tenengrad": metric_tenengrad(gray),
    }


def load_farm_weights_from_checkpoint(farm_model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("Unsupported checkpoint format.")

    farm_state = {}
    for k, v in state_dict.items():
        if k.startswith("farm."):
            farm_state[k[len("farm."):]] = v

    if len(farm_state) == 0:
        raise ValueError("No FARM weights found in checkpoint.")

    missing, unexpected = farm_model.load_state_dict(farm_state, strict=False)
    print("FARM weights loaded successfully.")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)


def save_before_after_figure(
    save_path,
    display_name,
    before_rgb,
    after_rgb,
    diff_map,
    metrics_before,
    metrics_after
):
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(before_rgb)
    ax1.set_title(display_name, fontsize=12)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(after_rgb)
    ax2.set_title(f"{display_name} After FARM", fontsize=12)
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(diff_map, cmap="hot")
    ax3.set_title("Difference Heatmap", fontsize=12)
    ax3.axis("off")

    text = (
        f"Saturation Ratio (↓): {metrics_before['saturation_ratio']:.4f} -> {metrics_after['saturation_ratio']:.4f}\n"
        f"Tenengrad (↑): {metrics_before['tenengrad']:.2f} -> {metrics_after['tenengrad']:.2f}"
    )
    plt.figtext(0.5, 0.02, text, ha="center", fontsize=11)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def save_summary_grid(save_path, results):
    """
    两行排列
    每个类别占两列：
    左：英文类别名
    右：英文类别名 + After FARM
    """
    n = len(results)
    if n == 0:
        return

    num_rows = 2
    num_per_row = math.ceil(n / 2)
    num_cols = num_per_row * 2

    fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))

    for idx, item in enumerate(results):
        row = idx // num_per_row
        col_group = idx % num_per_row

        ax_before = fig.add_subplot(num_rows, num_cols, row * num_cols + col_group * 2 + 1)
        ax_before.imshow(item["before_rgb"])
        ax_before.set_title(item["display_name"], fontsize=12)
        ax_before.axis("off")

        ax_after = fig.add_subplot(num_rows, num_cols, row * num_cols + col_group * 2 + 2)
        ax_after.imshow(item["after_rgb"])
        ax_after.set_title(f"{item['display_name']} After FARM", fontsize=12)
        ax_after.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    test_dir = "/FM-Vmamba/Vmamba/VMamba-main/test_image_256"
    checkpoint_path = "/FM-Vmamba/Vmamba/VMamba-main/best_vmamba_laryngeal.pth"
    output_dir = "./farm_before_after_results"
    input_size = 224

    desired_order = [
        "喉癌",
        "声带息肉",
        "声带白斑",
        "声带炎",
        "正常声带图片",
        "声带沟"
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "single"))

    remove_hidden_dirs(test_dir)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    farm = FARM(in_channels=3).to(device)
    farm.eval()

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please provide the trained checkpoint first."
        )

    load_farm_weights_from_checkpoint(farm, checkpoint_path, device)

    image_paths = load_image_paths(test_dir)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {test_dir}")

    summary_results = []

    with torch.no_grad():
        for img_path in image_paths:
            class_name_cn = get_class_name_from_path(img_path, test_dir)
            display_name = map_to_english_class_name(class_name_cn)
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            img_pil = Image.open(img_path).convert("RGB")
            x = transform(img_pil).unsqueeze(0).to(device)
            y = farm(x)

            before_rgb = tensor_to_rgb_image(x, mean, std)
            after_rgb = tensor_to_rgb_image(y, mean, std)
            diff_map = diff_heatmap(before_rgb, after_rgb)

            metrics_before = evaluate_simple_metrics(before_rgb)
            metrics_after = evaluate_simple_metrics(after_rgb)

            save_path = os.path.join(
                output_dir,
                "single",
                f"{display_name}_{img_name}_before_after.png"
            )
            save_before_after_figure(
                save_path=save_path,
                display_name=display_name,
                before_rgb=before_rgb,
                after_rgb=after_rgb,
                diff_map=diff_map,
                metrics_before=metrics_before,
                metrics_after=metrics_after
            )

            summary_results.append({
                "class_name_cn": class_name_cn,
                "display_name": display_name,
                "before_rgb": before_rgb,
                "after_rgb": after_rgb
            })

            print(f"Saved: {save_path}")

    def sort_key(item):
        if item["class_name_cn"] in desired_order:
            return desired_order.index(item["class_name_cn"])
        return len(desired_order)

    summary_results = sorted(summary_results, key=sort_key)

    grid_path = os.path.join(output_dir, "farm_before_after_summary.png")
    save_summary_grid(grid_path, summary_results)
    print(f"Saved: {grid_path}")


if __name__ == "__main__":
    main()