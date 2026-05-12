import argparse
import csv
import json
import os
import random
import shutil
import sys
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

try:
    from vmamba import VSSM
except ImportError as exc:
    raise ImportError(
        "没有找到本地 vmamba.py。请把本脚本放在 vmamba.py 同目录下，"
        "或者把 vmamba.py 所在目录加入 PYTHONPATH。"
    ) from exc


VARIANTS = OrderedDict(
    {
        "VMamba": {"use_farm": False, "use_mona": False},
        "VMamba+FARM": {"use_farm": True, "use_mona": False},
        "VMamba+Mona": {"use_farm": False, "use_mona": True},
        "VMamba+FARM+Mona": {"use_farm": True, "use_mona": True},
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description="FM-VMamba ablation study: VMamba / FARM / Mona")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/FM-Vmamba/Vmamba/VMamba-main/PreparedData/all",
        help="ImageFolder 数据目录，例如 PreparedData/all。目录下应为 各类别子文件夹。",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results_ablation_fm_vmamba",
        help="消融实验输出目录。",
    )
    parser.add_argument(
        "--vmamba-ckpt",
        type=str,
        default="/FM-Vmamba/Vmamba/VMamba-main/vssm1_tiny_0230s_ckpt_epoch_264.pth",
        help="VMamba 预训练权重路径。若不存在则从随机初始化开始。",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--train-policy",
        type=str,
        default="peft",
        choices=["peft", "full", "paper"],
        help="参数训练策略：peft / full / paper。论文消融建议先用 peft。",
    )

    parser.add_argument("--mona-bottleneck-dim", type=int, default=64)
    parser.add_argument("--mona-kernels", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--mona-dropout", type=float, default=0.0)

    # 为避免 Jupyter IOPub message rate exceeded，默认不显示 batch 级进度条。
    parser.add_argument("--show-progress", action="store_true", help="显示 tqdm 进度条。")
    parser.add_argument("--save-fold-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-confusion", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def remove_hidden_folders(root_dir: str):
    removed = False
    for current_root, dirnames, _ in os.walk(root_dir):
        for dirname in list(dirnames):
            if dirname.startswith("."):
                hidden_dir = os.path.join(current_root, dirname)
                try:
                    shutil.rmtree(hidden_dir)
                    print(f"Removed hidden folder: {hidden_dir}")
                    removed = True
                except Exception as exc:
                    print(f"Warning: failed to remove {hidden_dir}: {exc}")
    if not removed:
        print(f"No hidden folders found in {root_dir}.")


class FARM(nn.Module):
    """
    Fourier Adaptive Recovery Module.
    流程：FFT -> fftshift -> amplitude -> frequency controller -> mask -> iFFT -> residual add.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.freq_controller = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
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


class MonaAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int = 64,
        kernels=(3, 5, 7),
        adapter_dropout: float = 0.0,
        channel_first: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.kernels = tuple(kernels)
        self.channel_first = channel_first

        self.ln = nn.LayerNorm(in_dim)
        self.s1 = nn.Parameter(torch.ones(1))
        self.s2 = nn.Parameter(torch.ones(1))

        self.down_proj = nn.Linear(in_dim, bottleneck_dim)

        self.dw_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    bottleneck_dim,
                    bottleneck_dim,
                    kernel_size=k,
                    padding=k // 2,
                    groups=bottleneck_dim,
                    bias=True,
                )
                for k in self.kernels
            ]
        )

        self.pw_conv = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(adapter_dropout)
        self.up_proj = nn.Linear(bottleneck_dim, in_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

        for conv in self.dw_convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        nn.init.kaiming_normal_(self.pw_conv.weight, mode="fan_out", nonlinearity="relu")
        if self.pw_conv.bias is not None:
            nn.init.zeros_(self.pw_conv.bias)

        # 置零上投影，使 Adapter 初始时更接近恒等映射，训练更稳。
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"MonaAdapter expects 4D tensor, got {tuple(x.shape)}")

        if self.channel_first:
            x_bhwc = x.permute(0, 2, 3, 1).contiguous()
            restore = lambda t: t.permute(0, 3, 1, 2).contiguous()
        else:
            x_bhwc = x
            restore = lambda t: t

        residual = x_bhwc
        x_norm = self.s1 * self.ln(x_bhwc) + self.s2 * x_bhwc

        z = self.down_proj(x_norm)              # [B,H,W,d]
        z = z.permute(0, 3, 1, 2).contiguous() # [B,d,H,W]

        multi_scale = 0.0
        for conv in self.dw_convs:
            multi_scale = multi_scale + conv(z)
        multi_scale = multi_scale / len(self.dw_convs)

        z = z + multi_scale
        z = z + self.pw_conv(z)
        z = self.act(z)

        z = z.permute(0, 2, 3, 1).contiguous() # [B,H,W,d]
        z = self.dropout(z)
        z = self.up_proj(z)

        out = residual + z
        return restore(out)


def iter_vss_blocks(vssm_model):
    for stage in vssm_model.layers:
        if hasattr(stage, "blocks"):
            for block in stage.blocks:
                yield block


def inject_mona_into_vssm(vssm_model, bottleneck_dim=64, kernels=(3, 5, 7), adapter_dropout=0.0):
    channel_first = getattr(vssm_model, "channel_first", False)
    injected_count = 0

    for block in iter_vss_blocks(vssm_model):
        if getattr(block, "_mona_injected", False):
            continue

        if hasattr(block, "norm"):
            hidden_dim = int(block.norm.weight.numel())
        elif hasattr(block, "norm2"):
            hidden_dim = int(block.norm2.weight.numel())
        else:
            raise AttributeError("无法从 VSS block 中推断 hidden_dim：没有 norm/norm2。")

        if getattr(block, "ssm_branch", True):
            block.mona_after_ssm = MonaAdapter(
                in_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
                kernels=kernels,
                adapter_dropout=adapter_dropout,
                channel_first=channel_first,
            )

        if getattr(block, "mlp_branch", True):
            block.mona_after_mlp = MonaAdapter(
                in_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
                kernels=kernels,
                adapter_dropout=adapter_dropout,
                channel_first=channel_first,
            )

        def _forward_with_mona(self, input_tensor):
            x = input_tensor

            if self.ssm_branch:
                if self.post_norm:
                    x = x + self.drop_path(self.norm(self.op(x)))
                else:
                    x = x + self.drop_path(self.op(self.norm(x)))
                x = self.mona_after_ssm(x)

            if self.mlp_branch:
                if self.post_norm:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                x = self.mona_after_mlp(x)

            return x

        block._forward = _forward_with_mona.__get__(block, block.__class__)
        block._mona_injected = True
        injected_count += 1

    print(f"Mona injected into {injected_count} VSSBlocks.")
    return vssm_model


def build_vssm_backbone(num_classes):
    return VSSM(
        depths=[2, 2, 8, 2],
        dims=96,
        drop_path_rate=0.2,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=224,
    )


def load_vssm_pretrained(vssm_model, ckpt_path, strict=False):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print("[Warning] VMamba checkpoint not found. Training from scratch.")
        return False

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    filtered = {}
    for k, v in state_dict.items():
        k = k.removeprefix("module.")
        if "classifier.head" in k or k.startswith("head.") or ".head." in k:
            print(f"Skipping classifier parameter: {k}")
            continue
        filtered[k] = v

    incompatible = vssm_model.load_state_dict(filtered, strict=False)
    print(f"Loaded pretrained checkpoint from: {ckpt_path}")
    print(f"Missing keys: {incompatible.missing_keys}")
    print(f"Unexpected keys: {incompatible.unexpected_keys}")

    if strict:
        non_head_missing = [k for k in incompatible.missing_keys if "head" not in k]
        assert len(non_head_missing) == 0 and len(incompatible.unexpected_keys) == 0

    return True


class AblationFMVMamba(nn.Module):
    def __init__(self, num_classes, use_farm, use_mona, args):
        super().__init__()
        self.use_farm = use_farm
        self.use_mona = use_mona

        self.farm = FARM(in_channels=3) if use_farm else nn.Identity()
        self.vssm = build_vssm_backbone(num_classes=num_classes)
        self.has_pretrained = load_vssm_pretrained(self.vssm, args.vmamba_ckpt, strict=False)

        if use_mona:
            inject_mona_into_vssm(
                self.vssm,
                bottleneck_dim=args.mona_bottleneck_dim,
                kernels=tuple(args.mona_kernels),
                adapter_dropout=args.mona_dropout,
            )

    def forward(self, x):
        x = self.farm(x)
        x = self.vssm(x)
        return x


def set_train_policy(model, use_farm, use_mona, policy):
    if policy == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    if policy == "paper" and not use_mona:
        for p in model.parameters():
            p.requires_grad = True
        return

    # policy == "peft" 或 policy == "paper" 且 use_mona=True
    # 冻结 VMamba 主干；训练分类头；若有 FARM/Mona，则训练 FARM/Mona。
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "mona_after_ssm" in name or "mona_after_mlp" in name:
            p.requires_grad = True
        if name.startswith("farm.") and use_farm:
            p.requires_grad = True
        if "classifier" in name or ".head" in name or name.endswith("head.weight") or name.endswith("head.bias"):
            p.requires_grad = True


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable / total if total else 0.0
    return total, trainable, ratio


def build_dataset(data_dir, input_size):
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return datasets.ImageFolder(data_dir, transform=transform)


def validate_dataset(dataset, n_splits):
    targets = np.array(dataset.targets)
    counts = np.bincount(targets, minlength=len(dataset.classes))
    print("Class counts:")
    for cls_name, count in zip(dataset.classes, counts):
        print(f"  {cls_name}: {int(count)}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")
    if counts.min() < n_splits:
        raise ValueError(f"Each class must have at least {n_splits} images. Min count={counts.min()}")


def build_dataloaders(dataset, train_indices, val_indices, args, fold_seed):
    device = torch.device(args.device)
    generator = torch.Generator()
    generator.manual_seed(fold_seed)

    common_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker if args.num_workers > 0 else None,
        "generator": generator,
    }
    if args.num_workers > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **common_kwargs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    class_weights = None
    if args.use_class_weights:
        train_targets = np.array(dataset.targets)[train_indices]
        counts = np.bincount(train_targets, minlength=len(dataset.classes)).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32)

    return train_loader, val_loader, class_weights


class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def step(self, value):
        if self.best is None or value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp, show_progress):
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []
    amp_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    for images, labels in tqdm(loader, desc="Train", leave=False, disable=not show_progress):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_context:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

    return running_loss / len(loader.dataset), accuracy_score(all_labels, all_preds) * 100.0


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, show_progress, return_predictions=False):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    amp_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    with amp_context:
        for images, labels in tqdm(loader, desc="Eval", leave=False, disable=not show_progress):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs.float(), dim=1)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    metrics = {
        "loss": running_loss / len(loader.dataset),
        "acc": accuracy_score(all_labels, all_preds) * 100.0,
        "precision_macro": precision * 100.0,
        "recall_macro": recall * 100.0,
        "f1_macro": f1 * 100.0,
    }
    if return_predictions:
        return metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)
    return metrics


def compute_macro_auc_percent(y_true, y_prob, num_classes):
    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        score = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        return score * 100.0
    except Exception as exc:
        print(f"Warning: macro AUC unavailable: {exc}")
        return np.nan


def plot_training_curves(history, save_path):
    if not history:
        return
    epochs = [x["epoch"] for x in history]
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, [x["train_loss"] for x in history], label="Train Loss")
    ax1.plot(epochs, [x["val_loss"] for x in history], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs, [x["val_acc"] for x in history], label="Val Acc")
    ax2.plot(epochs, [x["val_f1_macro"] for x in history], label="Val Macro-F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Validation Metrics")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, class_names, save_path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary_df, results_dir):
    order_map = {name: i for i, name in enumerate(VARIANTS.keys())}
    df = summary_df.copy()
    df["order"] = df["variant"].map(order_map)
    df = df.sort_values("order")

    x = np.arange(len(df))
    metrics = [
        ("acc_mean", "acc_std", "Acc"),
        ("precision_macro_mean", "precision_macro_std", "Precision"),
        ("recall_macro_mean", "recall_macro_std", "Recall"),
        ("f1_macro_mean", "f1_macro_std", "F1"),
        ("auc_macro_mean", "auc_macro_std", "AUC"),
    ]

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    width = 0.15
    for i, (mean_col, std_col, label) in enumerate(metrics):
        ax.bar(x + (i - 2) * width, df[mean_col], width, yerr=df[std_col], capsize=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("Ablation Study: Performance Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=5, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig_ablation_metrics_bar.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    baseline = df[df["variant"] == "VMamba"].iloc[0]
    gain_cols = [
        ("acc_mean", "Acc Gain"),
        ("f1_macro_mean", "F1 Gain"),
        ("auc_macro_mean", "AUC Gain"),
    ]
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    width = 0.24
    for i, (col, label) in enumerate(gain_cols):
        ax.bar(x + (i - 1) * width, df[col] - baseline[col], width, label=label)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=15, ha="right")
    ax.set_ylabel("Gain over VMamba (%)")
    ax.set_title("Performance Gain over VMamba Baseline")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig_ablation_gain_over_vmamba.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    total_m = df["total_params_mean"] / 1e6
    trainable_m = df["trainable_params_mean"] / 1e6
    ax.bar(x - 0.18, total_m, width=0.36, label="Total Params")
    ax.bar(x + 0.18, trainable_m, width=0.36, label="Trainable Params")
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=15, ha="right")
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Parameter Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig_ablation_trainable_params.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    heat = df[["acc_mean", "precision_macro_mean", "recall_macro_mean", "f1_macro_mean", "auc_macro_mean"]].values
    fig = plt.figure(figsize=(9, 4.8))
    ax = fig.add_subplot(111)
    im = ax.imshow(heat, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(["Acc", "Precision", "Recall", "F1", "AUC"])
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["variant"])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Ablation Study Heatmap (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig_ablation_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_one_variant_fold(args, variant_name, variant_cfg, fold_idx, dataset, train_indices, val_indices, class_names):
    device = torch.device(args.device)
    safe_variant = variant_name.replace("+", "_plus_").replace(" ", "_")
    fold_dir = os.path.join(args.results_dir, safe_variant, f"fold_{fold_idx}")
    ensure_dir(fold_dir)

    print("\n" + "=" * 90)
    print(f"Variant={variant_name} | Fold={fold_idx}/{args.n_splits}")
    print("=" * 90)

    set_seed(args.seed + fold_idx)
    train_loader, val_loader, class_weights = build_dataloaders(
        dataset, train_indices, val_indices, args, fold_seed=args.seed + fold_idx
    )

    model = AblationFMVMamba(
        num_classes=len(class_names),
        use_farm=variant_cfg["use_farm"],
        use_mona=variant_cfg["use_mona"],
        args=args,
    ).to(device)

    set_train_policy(model, variant_cfg["use_farm"], variant_cfg["use_mona"], args.train_policy)
    total_params, trainable_params, trainable_ratio = count_parameters(model)
    print(f"Train policy: {args.train_policy}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_ratio:.4f}%")

    criterion_kwargs = {"label_smoothing": args.label_smoothing}
    if class_weights is not None:
        criterion_kwargs["weight"] = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(**criterion_kwargs)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters. Check train policy.")

    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    stopper = EarlyStopping(args.early_stop_patience, args.early_stop_min_delta)
    use_amp = device.type == "cuda"

    best_f1 = -1.0
    best_epoch = -1
    best_path = os.path.join(fold_dir, "best_model_state_dict.pth")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp, args.show_progress)
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp, args.show_progress)
        scheduler.step(val_metrics["f1_macro"])
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "lr": lr,
        }
        history.append(row)

        print(
            f"{variant_name} | Fold {fold_idx} | Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
            f"Val Loss={val_metrics['loss']:.4f} Acc={val_metrics['acc']:.2f}% "
            f"P={val_metrics['precision_macro']:.2f}% R={val_metrics['recall_macro']:.2f}% "
            f"F1={val_metrics['f1_macro']:.2f}% | LR={lr:.8f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"Saved best checkpoint: epoch={best_epoch}, F1={best_f1:.2f}%")

        if stopper.step(val_metrics["f1_macro"]):
            print("Early stopping triggered.")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(history).to_csv(os.path.join(fold_dir, "training_log.csv"), index=False, encoding="utf-8-sig")
    if args.save_fold_plots:
        plot_training_curves(history, os.path.join(fold_dir, "training_curves.png"))

    print(f"Loading best checkpoint: {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    final_metrics, y_true, y_pred, y_prob = evaluate(
        model, val_loader, criterion, device, use_amp, args.show_progress, return_predictions=True
    )
    auc_macro = compute_macro_auc_percent(y_true, y_prob, len(class_names))
    cm = confusion_matrix(y_true, y_pred)

    np.savetxt(os.path.join(fold_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    np.save(os.path.join(fold_dir, "y_true.npy"), y_true)
    np.save(os.path.join(fold_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(fold_dir, "y_prob.npy"), y_prob)

    if args.save_confusion:
        plot_confusion(cm, class_names, os.path.join(fold_dir, "confusion_matrix.png"))

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    with open(os.path.join(fold_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Variant: {variant_name}\n")
        f.write(f"Fold: {fold_idx}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Total params: {total_params}\n")
        f.write(f"Trainable params: {trainable_params}\n")
        f.write(f"Trainable ratio: {trainable_ratio:.6f}%\n")
        f.write(f"Loss: {final_metrics['loss']:.6f}\n")
        f.write(f"Accuracy: {final_metrics['acc']:.4f}%\n")
        f.write(f"Precision Macro: {final_metrics['precision_macro']:.4f}%\n")
        f.write(f"Recall Macro: {final_metrics['recall_macro']:.4f}%\n")
        f.write(f"F1 Macro: {final_metrics['f1_macro']:.4f}%\n")
        f.write(f"AUC Macro OVR: {auc_macro:.4f}%\n\n")
        f.write(report)

    result = {
        "variant": variant_name,
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "loss": final_metrics["loss"],
        "acc": final_metrics["acc"],
        "precision_macro": final_metrics["precision_macro"],
        "recall_macro": final_metrics["recall_macro"],
        "f1_macro": final_metrics["f1_macro"],
        "auc_macro": auc_macro,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
    }
    print("Fold result:", result)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize_results(results, args):
    ensure_dir(args.results_dir)
    df = pd.DataFrame(results)
    all_csv = os.path.join(args.results_dir, "ablation_all_folds.csv")
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    summary_rows = []
    for variant in VARIANTS.keys():
        sub = df[df["variant"] == variant]
        row = OrderedDict()
        row["variant"] = variant
        for col in [
            "loss",
            "acc",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "auc_macro",
            "total_params",
            "trainable_params",
            "trainable_ratio",
        ]:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.results_dir, "ablation_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    word_mean_rows, word_mean_std_rows = [], []
    for _, row in summary_df.iterrows():
        mean_row = OrderedDict()
        mean_std_row = OrderedDict()

        mean_row["模型组合"] = row["variant"]
        mean_row["Loss"] = f"{row['loss_mean']:.4f}"
        mean_row["准确率（Acc）"] = f"{row['acc_mean']:.2f}"
        mean_row["精确率（Precision）"] = f"{row['precision_macro_mean']:.2f}"
        mean_row["召回率（Recall）"] = f"{row['recall_macro_mean']:.2f}"
        mean_row["F1"] = f"{row['f1_macro_mean']:.2f}"
        mean_row["AUC"] = f"{row['auc_macro_mean']:.2f}"
        mean_row["总参数量（M）"] = f"{row['total_params_mean'] / 1e6:.2f}"
        mean_row["可训练参数量（M）"] = f"{row['trainable_params_mean'] / 1e6:.2f}"
        mean_row["可训练比例（%）"] = f"{row['trainable_ratio_mean']:.2f}"

        mean_std_row["模型组合"] = row["variant"]
        mean_std_row["Loss"] = f"{row['loss_mean']:.4f}±{row['loss_std']:.4f}"
        mean_std_row["准确率（Acc）"] = f"{row['acc_mean']:.2f}±{row['acc_std']:.2f}"
        mean_std_row["精确率（Precision）"] = f"{row['precision_macro_mean']:.2f}±{row['precision_macro_std']:.2f}"
        mean_std_row["召回率（Recall）"] = f"{row['recall_macro_mean']:.2f}±{row['recall_macro_std']:.2f}"
        mean_std_row["F1"] = f"{row['f1_macro_mean']:.2f}±{row['f1_macro_std']:.2f}"
        mean_std_row["AUC"] = f"{row['auc_macro_mean']:.2f}±{row['auc_macro_std']:.2f}"
        mean_std_row["总参数量（M）"] = f"{row['total_params_mean'] / 1e6:.2f}"
        mean_std_row["可训练参数量（M）"] = f"{row['trainable_params_mean'] / 1e6:.2f}"
        mean_std_row["可训练比例（%）"] = f"{row['trainable_ratio_mean']:.2f}"

        word_mean_rows.append(mean_row)
        word_mean_std_rows.append(mean_std_row)

    word_mean_df = pd.DataFrame(word_mean_rows)
    word_mean_std_df = pd.DataFrame(word_mean_std_rows)

    word_mean_csv = os.path.join(args.results_dir, "Table2_ablation_for_word_mean.csv")
    word_mean_std_csv = os.path.join(args.results_dir, "Table2_ablation_for_word_mean_std.csv")
    word_mean_xlsx = os.path.join(args.results_dir, "Table2_ablation_for_word_mean.xlsx")
    word_mean_std_xlsx = os.path.join(args.results_dir, "Table2_ablation_for_word_mean_std.xlsx")

    word_mean_df.to_csv(word_mean_csv, index=False, encoding="utf-8-sig")
    word_mean_std_df.to_csv(word_mean_std_csv, index=False, encoding="utf-8-sig")
    word_mean_df.to_excel(word_mean_xlsx, index=False)
    word_mean_std_df.to_excel(word_mean_std_xlsx, index=False)

    with open(os.path.join(args.results_dir, "ablation_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    plot_summary(summary_df, args.results_dir)

    print("\n" + "=" * 90)
    print("Ablation experiment finished.")
    print("=" * 90)
    print("\nAll fold results:")
    print(df)
    print("\nSummary:")
    print(summary_df)
    print("\nTable 2 mean:")
    print(word_mean_df)
    print("\nTable 2 mean±std:")
    print(word_mean_std_df)
    print("\nSaved files:")
    print(f"  {all_csv}")
    print(f"  {summary_csv}")
    print(f"  {word_mean_xlsx}")
    print(f"  {word_mean_std_xlsx}")
    print(f"  {os.path.join(args.results_dir, 'fig_ablation_metrics_bar.png')}")
    print(f"  {os.path.join(args.results_dir, 'fig_ablation_gain_over_vmamba.png')}")
    print(f"  {os.path.join(args.results_dir, 'fig_ablation_trainable_params.png')}")
    print(f"  {os.path.join(args.results_dir, 'fig_ablation_heatmap.png')}")

    return df, summary_df, word_mean_df, word_mean_std_df


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.results_dir)
    remove_hidden_folders(args.data_dir)

    dataset = build_dataset(args.data_dir, args.input_size)
    validate_dataset(dataset, args.n_splits)
    class_names = list(dataset.classes)
    targets = np.array(dataset.targets)

    print("\n" + "=" * 90)
    print("Ablation Dataset Information")
    print("=" * 90)
    print(f"Classes: {dataset.class_to_idx}")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Variants: {list(VARIANTS.keys())}")
    print(f"Train policy: {args.train_policy}")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    x_dummy = np.zeros(len(targets))
    splits = list(skf.split(x_dummy, targets))

    all_results = []
    for variant_name, variant_cfg in VARIANTS.items():
        for fold_idx, (train_indices, val_indices) in enumerate(splits, start=1):
            result = train_one_variant_fold(
                args=args,
                variant_name=variant_name,
                variant_cfg=variant_cfg,
                fold_idx=fold_idx,
                dataset=dataset,
                train_indices=train_indices,
                val_indices=val_indices,
                class_names=class_names,
            )
            all_results.append(result)
            pd.DataFrame(all_results).to_csv(
                os.path.join(args.results_dir, "ablation_all_folds_partial.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    summarize_results(all_results, args)


if __name__ == "__main__":
    main()
