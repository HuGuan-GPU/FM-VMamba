import argparse
import csv
import json
import os
import random
import shutil
import sys
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
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))


DEFAULT_MODELS = [
    "resnet50",
    "mobilenet_v2",
    "efficientnet_v2_s",
    "swin_t",
    "mobilevit_s",
    "vmamba",
]


def parse_args():
    parser = argparse.ArgumentParser(description="5-fold comparison experiments for laryngeal lesion classification.")
    parser.add_argument("--data-dir", type=str, required=True, help="ImageFolder root, e.g. PreparedData/all")
    parser.add_argument("--results-dir", type=str, default="./results_comparison_5fold")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights when available.")
    parser.add_argument("--vmamba-ckpt", type=str, default="", help="Optional pretrained VMamba checkpoint.")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed):
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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def remove_hidden_folders(root_dir):
    for current_root, dirnames, _ in os.walk(root_dir):
        for dirname in list(dirnames):
            if dirname.startswith("."):
                hidden_dir = os.path.join(current_root, dirname)
                try:
                    shutil.rmtree(hidden_dir)
                    print(f"Removed hidden folder: {hidden_dir}")
                except Exception as exc:
                    print(f"Warning: failed to remove {hidden_dir}: {exc}")


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
    print("Class distribution:")
    for name, count in zip(dataset.classes, counts):
        print(f"  {name}: {int(count)}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")
    if counts.min() < n_splits:
        raise ValueError(f"Each class must have at least {n_splits} images. Min class count: {counts.min()}")


def build_dataloaders(dataset, train_indices, val_indices, batch_size, num_workers, device, seed, use_class_weights):
    generator = torch.Generator()
    generator.manual_seed(seed)

    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": generator,
    }
    if num_workers > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **common_kwargs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    class_weights = None
    if use_class_weights:
        train_targets = np.array(dataset.targets)[train_indices]
        counts = np.bincount(train_targets, minlength=len(dataset.classes)).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32)

    return train_loader, val_loader, class_weights


def replace_linear_head(module, num_classes):
    in_features = module.in_features
    return nn.Linear(in_features, num_classes)


def create_torchvision_model(model_name, num_classes, pretrained):
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = replace_linear_head(model.fc, num_classes)
        return model

    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[-1] = replace_linear_head(model.classifier[-1], num_classes)
        return model

    if model_name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[-1] = replace_linear_head(model.classifier[-1], num_classes)
        return model

    if model_name == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = replace_linear_head(model.head, num_classes)
        return model

    raise ValueError(f"Unsupported torchvision model: {model_name}")


def create_mobilevit_model(num_classes, pretrained):
    try:
        import timm
    except ImportError as exc:
        raise ImportError("mobilevit_s requires timm. Install it or remove mobilevit_s from --models.") from exc
    return timm.create_model("mobilevit_s", pretrained=pretrained, num_classes=num_classes)


def create_vmamba_model(num_classes, ckpt_path=""):
    try:
        from vmamba import VSSM
    except ImportError as exc:
        raise ImportError("vmamba baseline requires local vmamba.py in the same project.") from exc

    model = VSSM(
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

    if ckpt_path and os.path.isfile(ckpt_path):
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
            if "classifier.head" in k or k.startswith("head."):
                continue
            filtered[k] = v
        incompatible = model.load_state_dict(filtered, strict=False)
        print(f"Loaded VMamba checkpoint: {ckpt_path}")
        print(f"Missing keys: {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")
    else:
        print("VMamba checkpoint not provided or not found. Training VMamba from scratch.")

    return model


def create_model(model_name, num_classes, pretrained, vmamba_ckpt):
    if model_name in {"resnet50", "mobilenet_v2", "efficientnet_v2_s", "swin_t"}:
        return create_torchvision_model(model_name, num_classes, pretrained)
    if model_name == "mobilevit_s":
        return create_mobilevit_model(num_classes, pretrained)
    if model_name == "vmamba":
        return create_vmamba_model(num_classes, vmamba_ckpt)
    raise ValueError(f"Unknown model name: {model_name}")


class EarlyStopping:
    def __init__(self, patience=12, min_delta=0.0):
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


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp):
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []
    amp_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    for images, labels in tqdm(loader, desc="Train", leave=False):
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
def evaluate(model, loader, criterion, device, use_amp, return_predictions=False):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    amp_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    with amp_context:
        for images, labels in tqdm(loader, desc="Eval", leave=False):
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


def plot_curves(history, save_path):
    if not history:
        return
    epochs = [x["epoch"] for x in history]
    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(epochs, [x["train_loss"] for x in history], label="Train Loss")
    ax1.plot(epochs, [x["val_loss"] for x in history], label="Val Loss")
    ax1.set_title("Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(epochs, [x["train_acc"] for x in history], label="Train Acc")
    ax2.plot(epochs, [x["val_acc"] for x in history], label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(epochs, [x["val_f1_macro"] for x in history], label="Val Macro-F1")
    ax3.set_title("Macro-F1")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, class_names, save_path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(im)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, y_prob, class_names, save_path):
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    auc_dict = {}

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for i, name in enumerate(class_names):
        positives = np.sum(y_true_bin[:, i])
        negatives = len(y_true_bin) - positives
        if positives == 0 or negatives == 0:
            auc_dict[name] = np.nan
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        score = auc(fpr, tpr)
        auc_dict[name] = score
        ax.plot(fpr, tpr, lw=2, label=f"{name} AUC={score:.4f}")

    ax.plot([0, 1], [0, 1], "--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-class ROC")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    try:
        macro_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
    except Exception as exc:
        print(f"Warning: macro AUC unavailable: {exc}")
        macro_auc = np.nan

    return auc_dict, macro_auc


def save_history(history, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "val_precision_macro",
                "val_recall_macro",
                "val_f1_macro",
                "lr",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def train_one_model_fold(args, model_name, fold_idx, dataset, train_indices, val_indices, class_names):
    device = torch.device(args.device)
    fold_dir = os.path.join(args.results_dir, model_name, f"fold_{fold_idx}")
    ensure_dir(fold_dir)

    train_loader, val_loader, class_weights = build_dataloaders(
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed + fold_idx,
        use_class_weights=args.use_class_weights,
    )

    model = create_model(model_name, len(class_names), args.pretrained, args.vmamba_ckpt).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"{model_name} fold {fold_idx}: total params={total_params:,}, trainable={trainable_params:,}")

    criterion_kwargs = {"label_smoothing": args.label_smoothing}
    if class_weights is not None:
        criterion_kwargs["weight"] = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(**criterion_kwargs)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    stopper = EarlyStopping(patience=args.early_stop_patience)
    use_amp = device.type == "cuda"

    best_f1 = -1.0
    best_path = os.path.join(fold_dir, "best_checkpoint.pth")
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nModel={model_name} | Fold={fold_idx}/{args.n_splits} | Epoch={epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp)
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
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
            f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
            f"Val Loss={val_metrics['loss']:.4f} Acc={val_metrics['acc']:.2f}% "
            f"P={val_metrics['precision_macro']:.2f}% R={val_metrics['recall_macro']:.2f}% "
            f"F1={val_metrics['f1_macro']:.2f}% | LR={lr:.8f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_f1_macro": best_f1,
                    "class_names": class_names,
                    "args": vars(args),
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                },
                best_path,
            )
            print(f"Saved best checkpoint: F1={best_f1:.2f}%")

        if stopper.step(val_metrics["f1_macro"]):
            print("Early stopping triggered.")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_history(history, os.path.join(fold_dir, "training_log.csv"))
    plot_curves(history, os.path.join(fold_dir, "training_curves.png"))

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics, y_true, y_pred, y_prob = evaluate(model, val_loader, criterion, device, use_amp, return_predictions=True)

    cm = confusion_matrix(y_true, y_pred)
    auc_dict, macro_auc = plot_roc(y_true, y_prob, class_names, os.path.join(fold_dir, "multiclass_roc.png"))
    plot_confusion(cm, class_names, os.path.join(fold_dir, "confusion_matrix.png"))
    np.savetxt(os.path.join(fold_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    np.save(os.path.join(fold_dir, "y_true.npy"), y_true)
    np.save(os.path.join(fold_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(fold_dir, "y_prob.npy"), y_prob)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    with open(os.path.join(fold_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Fold: {fold_idx}\n")
        f.write(f"Best epoch: {checkpoint['epoch']}\n")
        f.write(f"Total params: {total_params}\n")
        f.write(f"Trainable params: {trainable_params}\n")
        f.write(f"Loss: {final_metrics['loss']:.6f}\n")
        f.write(f"Accuracy: {final_metrics['acc']:.4f}%\n")
        f.write(f"Precision Macro: {final_metrics['precision_macro']:.4f}%\n")
        f.write(f"Recall Macro: {final_metrics['recall_macro']:.4f}%\n")
        f.write(f"F1 Macro: {final_metrics['f1_macro']:.4f}%\n")
        f.write(f"AUC Macro OVR: {macro_auc:.6f}\n\n")
        f.write("Per-class AUC:\n")
        for cls, value in auc_dict.items():
            f.write(f"{cls}: {value:.6f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

    return {
        "model": model_name,
        "fold": fold_idx,
        "best_epoch": checkpoint["epoch"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "loss": final_metrics["loss"],
        "acc": final_metrics["acc"],
        "precision_macro": final_metrics["precision_macro"],
        "recall_macro": final_metrics["recall_macro"],
        "f1_macro": final_metrics["f1_macro"],
        "auc_macro": macro_auc,
    }


def summarize_results(results, results_dir):
    df = pd.DataFrame(results)
    all_csv = os.path.join(results_dir, "all_model_fold_results.csv")
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby("model")
        .agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            precision_mean=("precision_macro", "mean"),
            precision_std=("precision_macro", "std"),
            recall_mean=("recall_macro", "mean"),
            recall_std=("recall_macro", "std"),
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            auc_mean=("auc_macro", "mean"),
            auc_std=("auc_macro", "std"),
            params=("total_params", "first"),
        )
        .reset_index()
    )
    summary_csv = os.path.join(results_dir, "comparison_summary.csv")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with open(os.path.join(results_dir, "comparison_summary.txt"), "w", encoding="utf-8") as f:
        f.write("================ Comparison Summary ================\n\n")
        for _, row in summary.iterrows():
            f.write(
                f"{row['model']}: "
                f"Acc={row['acc_mean']:.4f}±{row['acc_std']:.4f}, "
                f"Precision={row['precision_mean']:.4f}±{row['precision_std']:.4f}, "
                f"Recall={row['recall_mean']:.4f}±{row['recall_std']:.4f}, "
                f"F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}, "
                f"AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f}, "
                f"Params={int(row['params'])}\n"
            )

    print(f"\nSaved all fold results: {all_csv}")
    print(f"Saved model summary: {summary_csv}")
    print(summary)


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.results_dir)
    remove_hidden_folders(args.data_dir)

    with open(os.path.join(args.results_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    dataset = build_dataset(args.data_dir, args.input_size)
    validate_dataset(dataset, args.n_splits)
    class_names = list(dataset.classes)
    targets = np.array(dataset.targets)

    print(f"Classes: {dataset.class_to_idx}")
    print(f"Total samples: {len(dataset)}")
    print(f"Models: {args.models}")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    x_dummy = np.zeros(len(targets))
    splits = list(skf.split(x_dummy, targets))

    all_results = []
    for model_name in args.models:
        for fold_idx, (train_indices, val_indices) in enumerate(splits, start=1):
            result = train_one_model_fold(args, model_name, fold_idx, dataset, train_indices, val_indices, class_names)
            all_results.append(result)

            pd.DataFrame(all_results).to_csv(
                os.path.join(args.results_dir, "all_model_fold_results_partial.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    summarize_results(all_results, args.results_dir)


if __name__ == "__main__":
    main()
