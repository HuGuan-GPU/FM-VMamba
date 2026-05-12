import os
import sys
import csv
import json
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold


if torch.cuda.is_available():
    torch.cuda.empty_cache()

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

try:
    from vmamba import VSSM
    print("Successfully imported VSSM from local vmamba.py")
except ImportError as e:
    print(f"Failed to import vmamba.py. Error: {e}")
    sys.exit(1)


class FARM(nn.Module):
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
        self.kernels = kernels
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
                for k in kernels
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

        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"MonaAdapter expects 4D tensor, got {tuple(x.shape)}")

        if self.channel_first:
            x_bhwc = x.permute(0, 2, 3, 1).contiguous()

            def restore(t):
                return t.permute(0, 3, 1, 2).contiguous()
        else:
            x_bhwc = x

            def restore(t):
                return t

        residual = x_bhwc
        x_norm = self.s1 * self.ln(x_bhwc) + self.s2 * x_bhwc

        z = self.down_proj(x_norm)
        z = z.permute(0, 3, 1, 2).contiguous()

        multi_scale = sum(conv(z) for conv in self.dw_convs) / len(self.dw_convs)
        z = z + multi_scale
        z = z + self.pw_conv(z)
        z = self.act(z)

        z = z.permute(0, 2, 3, 1).contiguous()
        z = self.dropout(z)
        z = self.up_proj(z)

        return restore(residual + z)


def iter_vss_blocks(vssm_model):
    for stage in vssm_model.layers:
        if hasattr(stage, "blocks"):
            for block in stage.blocks:
                yield block


def infer_block_hidden_dim(block):
    if hasattr(block, "norm") and hasattr(block.norm, "weight"):
        return int(block.norm.weight.numel())
    if hasattr(block, "norm2") and hasattr(block.norm2, "weight"):
        return int(block.norm2.weight.numel())
    raise AttributeError("Cannot infer VSS block hidden dim from norm/norm2.")


def inject_mona_into_vssm(vssm_model, bottleneck_dim=64, kernels=(3, 5, 7), adapter_dropout=0.0):
    channel_first = bool(getattr(vssm_model, "channel_first", False))
    injected_count = 0

    for block in iter_vss_blocks(vssm_model):
        if getattr(block, "_mona_injected", False):
            continue

        hidden_dim = infer_block_hidden_dim(block)
        ssm_branch = bool(getattr(block, "ssm_branch", True))
        mlp_branch = bool(getattr(block, "mlp_branch", True))

        if ssm_branch:
            block.mona_after_ssm = MonaAdapter(
                in_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
                kernels=kernels,
                adapter_dropout=adapter_dropout,
                channel_first=channel_first,
            )

        if mlp_branch:
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

        # VMamba VSSBlock.forward normally dispatches to _forward, preserving checkpoint logic.
        block._forward = _forward_with_mona.__get__(block, block.__class__)
        block._mona_injected = True
        injected_count += 1

    print(f"Mona injected into {injected_count} VSSBlocks.")
    return vssm_model


class FM_VMamba_Mona(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained_vssm_path=None,
        strict_load=False,
        mona_bottleneck_dim=64,
        mona_kernels=(3, 5, 7),
        mona_dropout=0.0,
        freeze_backbone_for_peft=True,
        train_farm=True,
    ):
        super().__init__()
        self.farm = FARM(in_channels=3)

        self.vssm = VSSM(
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

        has_pretrained = False
        if pretrained_vssm_path and os.path.isfile(pretrained_vssm_path):
            has_pretrained = True
            self.load_vssm_pretrained(pretrained_vssm_path, strict=strict_load)
        else:
            print("Warning: pretrained_vssm_path is not provided or does not exist.")

        inject_mona_into_vssm(
            self.vssm,
            bottleneck_dim=mona_bottleneck_dim,
            kernels=mona_kernels,
            adapter_dropout=mona_dropout,
        )

        if freeze_backbone_for_peft and not has_pretrained:
            print("Warning: no pretrained backbone found; PEFT freezing is disabled automatically.")
            freeze_backbone_for_peft = False

        if freeze_backbone_for_peft:
            self.freeze_for_mona_tuning(train_farm=train_farm)

    def load_vssm_pretrained(self, ckpt_path, strict=False):
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

        filtered_dict = {}
        for k, v in state_dict.items():
            clean_key = k.removeprefix("module.")
            if "classifier.head" in clean_key:
                print(f"Skipping classifier parameter: {clean_key}")
                continue
            filtered_dict[clean_key] = v

        incompatible = self.vssm.load_state_dict(filtered_dict, strict=False)
        print(f"Loaded pretrained checkpoint from: {ckpt_path}")
        print(f"Missing keys: {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")

        if strict:
            non_head_missing = [k for k in incompatible.missing_keys if "classifier.head" not in k]
            assert len(non_head_missing) == 0 and len(incompatible.unexpected_keys) == 0

    def freeze_for_mona_tuning(self, train_farm=True):
        for p in self.parameters():
            p.requires_grad = False

        for name, p in self.named_parameters():
            if "mona_after_ssm" in name or "mona_after_mlp" in name:
                p.requires_grad = True

        for p in self.vssm.classifier.parameters():
            p.requires_grad = True

        if train_farm:
            for p in self.farm.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.vssm(self.farm(x))



CONFIG = {
    "all_data_dir": "/FM-Vmamba/Vmamba/VMamba-main/PreparedData/all",
    "pretrained_vssm_path": "/FM-Vmamba/Vmamba/VMamba-main/vssm1_tiny_0230s_ckpt_epoch_264.pth",
    "strict_load": False,
    "mona_bottleneck_dim": 64,
    "mona_kernels": (3, 5, 7),
    "mona_dropout": 0.0,
    "freeze_backbone_for_peft": True,
    "train_farm_in_peft": True,
    "batch_size": 24,
    "num_workers": 4,
    "input_size": 224,
    "n_splits": 5,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "num_epochs": 300,
    "label_smoothing": 0.0,
    "use_class_weights": True,
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "min_lr": 1e-7,
    "early_stopping_patience": 8,
    "early_stopping_min_delta": 0.0,
    "seed": 42,
    "results_dir": "./results_fm_vmamba_mona_5fold",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def remove_hidden_folders(root_dir: str):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {root_dir}")

    removed = False
    for current_root, dirnames, _ in os.walk(root_dir):
        for d in list(dirnames):
            if d.startswith("."):
                hidden_dir = os.path.join(current_root, d)
                try:
                    shutil.rmtree(hidden_dir)
                    print(f"Removed hidden folder: {hidden_dir}")
                    removed = True
                except Exception as exc:
                    print(f"Warning: failed to remove hidden folder {hidden_dir}: {exc}")
    if not removed:
        print(f"No hidden folders found in {root_dir}.")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable / total if total else 0.0
    return total, trainable, ratio


def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


class EarlyStopping:
    def __init__(self, patience=8, mode="max", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "max":
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            self.should_stop = self.counter >= self.patience

        return self.should_stop


def build_full_dataset(all_dir, input_size):
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return datasets.ImageFolder(all_dir, transform=transform)


def validate_dataset_for_kfold(dataset, n_splits):
    targets = np.array(dataset.targets)
    if len(targets) == 0:
        raise ValueError("Dataset is empty.")

    counts = np.bincount(targets, minlength=len(dataset.classes))
    min_count = counts.min()
    print("Class counts:")
    for class_name, count in zip(dataset.classes, counts):
        print(f"  {class_name}: {int(count)}")

    if min_count < n_splits:
        raise ValueError(
            f"Each class must have at least n_splits samples. "
            f"min class count={int(min_count)}, n_splits={n_splits}"
        )


def build_fold_dataloaders(dataset, train_indices, val_indices, batch_size, num_workers, device, use_class_weights=True, seed=42):
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = device.type == "cuda"
    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": generator,
    }
    if num_workers > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    class_weights = None
    if use_class_weights:
        all_targets = np.array(dataset.targets)
        train_targets = all_targets[train_indices]
        num_classes = len(dataset.classes)
        counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32)

    return train_loader, val_loader, class_weights


def save_training_log(history, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_precision_macro",
            "val_recall_macro",
            "val_f1_macro",
            "lr",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(history, save_path):
    if not history:
        return

    epochs = [x["epoch"] for x in history]
    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(epochs, [x["train_loss"] for x in history], label="Train Loss")
    ax1.plot(epochs, [x["val_loss"] for x in history], label="Val Loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(epochs, [x["train_acc"] for x in history], label="Train Accuracy")
    ax2.plot(epochs, [x["val_acc"] for x in history], label="Val Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(epochs, [x["val_f1_macro"] for x in history], label="Val Macro-F1")
    ax3.set_title("Val Macro-F1 Curve")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Macro-F1 (%)")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, save_path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multiclass_roc(y_true, y_prob, class_names, save_path):
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    auc_dict = {}

    for i in range(num_classes):
        positives = np.sum(y_true_bin[:, i])
        negatives = len(y_true_bin) - positives
        if positives == 0 or negatives == 0:
            auc_dict[class_names[i]] = np.nan
            continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_dict[class_names[i]] = roc_auc
        ax.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC={roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("Multi-class ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    macro_auc = np.nan
    try:
        macro_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
    except Exception as exc:
        print(f"Warning: macro AUC is unavailable: {exc}")

    return auc_dict, macro_auc


def build_optimizer(model, base_lr, weight_decay):
    mona_params, farm_params, head_params, other_params = [], [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "mona_after_ssm" in name or "mona_after_mlp" in name:
            mona_params.append(p)
        elif name.startswith("farm."):
            farm_params.append(p)
        elif name.startswith("vssm.classifier."):
            head_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if mona_params:
        param_groups.append({"params": mona_params, "lr": base_lr, "name": "mona"})
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr, "name": "classifier"})
    if farm_params:
        param_groups.append({"params": farm_params, "lr": base_lr * 0.5, "name": "farm"})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "other"})

    if not param_groups:
        raise ValueError("No trainable parameters found.")

    return optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=False):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    trainable_params = get_trainable_params(model)

    progress_bar = tqdm(loader, desc="Train", leave=False)
    amp_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp_context:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        batch_acc = accuracy_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc * 100:.2f}%")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device, use_amp=False, return_predictions=False):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    progress_bar = tqdm(loader, desc="Eval", leave=False)
    amp_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    with amp_context:
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Cast to float32 before numpy conversion; bfloat16 tensors cannot always convert to numpy.
            probs = torch.softmax(outputs.float(), dim=1)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100.0
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    metrics = {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "precision_macro": precision * 100.0,
        "recall_macro": recall * 100.0,
        "f1_macro": f1 * 100.0,
    }

    if return_predictions:
        return metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)
    return metrics



def train_one_fold(fold_idx, dataset, train_indices, val_indices, class_names):
    print(f"\n================ Fold {fold_idx}/{CONFIG['n_splits']} ================\n")
    fold_dir = os.path.join(CONFIG["results_dir"], f"fold_{fold_idx}")
    ensure_dir(fold_dir)

    train_loader, val_loader, class_weights = build_fold_dataloaders(
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        device=CONFIG["device"],
        use_class_weights=CONFIG["use_class_weights"],
        seed=CONFIG["seed"] + fold_idx,
    )

    model = FM_VMamba_Mona(
        num_classes=len(class_names),
        pretrained_vssm_path=CONFIG["pretrained_vssm_path"],
        strict_load=CONFIG["strict_load"],
        mona_bottleneck_dim=CONFIG["mona_bottleneck_dim"],
        mona_kernels=CONFIG["mona_kernels"],
        mona_dropout=CONFIG["mona_dropout"],
        freeze_backbone_for_peft=CONFIG["freeze_backbone_for_peft"],
        train_farm=CONFIG["train_farm_in_peft"],
    ).to(CONFIG["device"])

    total_params, trainable_params, ratio = count_parameters(model)
    print(f"Fold {fold_idx} Total params: {total_params:,}")
    print(f"Fold {fold_idx} Trainable params: {trainable_params:,}")
    print(f"Fold {fold_idx} Trainable ratio: {ratio:.4f}%")

    save_json(
        {
            "fold": fold_idx,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_ratio_percent": ratio,
            "class_names": class_names,
            "config": CONFIG,
        },
        os.path.join(fold_dir, "run_config.json"),
    )

    criterion_kwargs = {"label_smoothing": CONFIG["label_smoothing"]}
    if class_weights is not None:
        criterion_kwargs["weight"] = class_weights.to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss(**criterion_kwargs)

    optimizer = build_optimizer(model=model, base_lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CONFIG["scheduler_factor"],
        patience=CONFIG["scheduler_patience"],
        min_lr=CONFIG["min_lr"],
    )
    early_stopper = EarlyStopping(
        patience=CONFIG["early_stopping_patience"],
        mode="max",
        min_delta=CONFIG["early_stopping_min_delta"],
    )

    use_amp = CONFIG["device"].type == "cuda"
    best_metric = -1.0
    history = []
    best_ckpt_path = os.path.join(fold_dir, "best_checkpoint.pth")

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nFold {fold_idx} | Epoch [{epoch + 1}/{CONFIG['num_epochs']}]")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=CONFIG["device"],
            use_amp=use_amp,
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=CONFIG["device"],
            use_amp=use_amp,
            return_predictions=False,
        )

        current_metric = val_metrics["f1_macro"]
        scheduler.step(current_metric)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.2f}% | "
            f"Val Precision(Macro): {val_metrics['precision_macro']:.2f}% | "
            f"Val Recall(Macro): {val_metrics['recall_macro']:.2f}% | "
            f"Val F1(Macro): {val_metrics['f1_macro']:.2f}% | LR: {current_lr:.8f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_precision_macro": val_metrics["precision_macro"],
                "val_recall_macro": val_metrics["recall_macro"],
                "val_f1_macro": val_metrics["f1_macro"],
                "lr": current_lr,
            }
        )

        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_f1_macro": best_metric,
                    "class_names": class_names,
                    "config": CONFIG,
                },
                best_ckpt_path,
            )
            print(f"Fold {fold_idx}: New best checkpoint saved. Best Val Macro-F1: {best_metric:.2f}%")

        stop = early_stopper.step(current_metric)
        print(f"Fold {fold_idx} EarlyStopping Counter: {early_stopper.counter}/{CONFIG['early_stopping_patience']}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if stop:
            print(f"Fold {fold_idx}: Early stopping triggered.")
            break

    save_training_log(history, os.path.join(fold_dir, "training_log.csv"))
    plot_training_curves(history, os.path.join(fold_dir, "training_curves.png"))

    if not os.path.isfile(best_ckpt_path):
        raise FileNotFoundError(f"No best checkpoint was saved for fold {fold_idx}.")

    print(f"\nFold {fold_idx}: Loading best checkpoint for final evaluation...")
    checkpoint = torch.load(best_ckpt_path, map_location=CONFIG["device"])
    model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics, y_true, y_pred, y_prob = evaluate_one_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=CONFIG["device"],
        use_amp=use_amp,
        return_predictions=True,
    )

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    auc_dict, macro_auc = plot_multiclass_roc(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        save_path=os.path.join(fold_dir, "multiclass_roc.png"),
    )
    plot_confusion_matrix(cm, class_names, os.path.join(fold_dir, "confusion_matrix.png"))

    with open(os.path.join(fold_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("=============== Fold Final Evaluation ===============\n")
        f.write(f"Loss: {final_metrics['loss']:.6f}\n")
        f.write(f"Accuracy: {final_metrics['acc']:.4f}%\n")
        f.write(f"Precision (Macro): {final_metrics['precision_macro']:.4f}%\n")
        f.write(f"Recall (Macro): {final_metrics['recall_macro']:.4f}%\n")
        f.write(f"F1-score (Macro): {final_metrics['f1_macro']:.4f}%\n")
        f.write(f"AUC (Macro, OVR): {macro_auc:.6f}\n")
        f.write("\nPer-class AUC:\n")
        for cls_name, auc_value in auc_dict.items():
            f.write(f"{cls_name}: {auc_value:.6f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

    np.savetxt(os.path.join(fold_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    np.save(os.path.join(fold_dir, "y_true.npy"), y_true)
    np.save(os.path.join(fold_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(fold_dir, "y_prob.npy"), y_prob)

    fold_result = {
        "fold": fold_idx,
        "loss": final_metrics["loss"],
        "acc": final_metrics["acc"],
        "precision_macro": final_metrics["precision_macro"],
        "recall_macro": final_metrics["recall_macro"],
        "f1_macro": final_metrics["f1_macro"],
        "auc_macro": macro_auc,
        "best_epoch": checkpoint["epoch"],
        "best_val_f1_macro": checkpoint["best_val_f1_macro"],
    }

    print(f"\nFold {fold_idx} result: {fold_result}")
    return fold_result


def run_five_fold_cross_validation():
    set_seed(CONFIG["seed"])
    ensure_dir(CONFIG["results_dir"])

    remove_hidden_folders(CONFIG["all_data_dir"])
    dataset = build_full_dataset(all_dir=CONFIG["all_data_dir"], input_size=CONFIG["input_size"])
    validate_dataset_for_kfold(dataset, CONFIG["n_splits"])

    targets = np.array(dataset.targets)
    class_names = list(dataset.classes)
    num_classes = len(class_names)

    print(f"Detected classes: {dataset.class_to_idx}")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {num_classes}")

    skf = StratifiedKFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=CONFIG["seed"])
    fold_results = []
    x_dummy = np.zeros(len(targets))

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(x_dummy, targets), start=1):
        fold_result = train_one_fold(
            fold_idx=fold_idx,
            dataset=dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            class_names=class_names,
        )
        fold_results.append(fold_result)

    df = pd.DataFrame(fold_results)
    csv_path = os.path.join(CONFIG["results_dir"], "five_fold_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary_txt_path = os.path.join(CONFIG["results_dir"], "five_fold_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("================ 5-Fold Cross Validation Summary ================\n\n")
        for metric in ["loss", "acc", "precision_macro", "recall_macro", "f1_macro", "auc_macro"]:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            line = f"{metric}: {mean_val:.4f} ± {std_val:.4f}"
            print(line)
            f.write(line + "\n")

    print("\nPer-fold results:")
    print(df)
    print("\nAll fold results saved to:")
    print(CONFIG["results_dir"])
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_txt_path}")


if __name__ == "__main__":
    run_five_fold_cross_validation()
