import os
import cv2
import json
import math
import shutil
import random
import numpy as np
import albumentations as A

from glob import glob
from tqdm import tqdm
from packaging import version



INPUT_DATA_DIR = "/root/FM-Vmamba/Vmamba/OriginalData"
OUTPUT_ROOT_DIR = "/root/FM-Vmamba/Vmamba/PreparedData"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = [
    "*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff",
    "*.JPG", "*.JPEG", "*.PNG", "*.TIF", "*.TIFF"
]


BALANCE_MODE = "adaptive"

FIXED_TARGET_COUNT = 900

MAX_AUG_RATIO = 8

ALLOW_HORIZONTAL_FLIP = True

SUMMARY_JSON = "split_summary.json"
CLASS_DISTRIBUTION_JSON = "class_distribution.json"


def build_augmentor():
    alb_ver = version.parse(A.__version__)

    affine = A.Affine(
        scale=(0.92, 1.08),
        translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
        rotate=(-10, 10),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.60
    )

    if alb_ver >= version.parse("2.0.0"):
        noise_aug = A.GaussNoise(
            std_range=(0.01, 0.02),
            mean_range=(0.0, 0.0),
            p=1.0
        )
    else:
        noise_aug = A.GaussNoise(
            var_limit=(5.0, 20.0),
            p=1.0
        )

    augmentor = A.Compose([
        A.HorizontalFlip(p=0.5 if ALLOW_HORIZONTAL_FLIP else 0.0),

        affine,

        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=1.0
            ),
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=1.0
            ),
        ], p=0.25),

        A.OneOf([
            noise_aug,
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.15),
    ])

    return augmentor


augmentor = build_augmentor()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def reset_output_dir(path: str):
    if os.path.exists(path):
        print(f"Removing old output directory: {path}")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def remove_hidden_folders(root_dir: str):
    removed = 0
    for current_root, dirnames, _ in os.walk(root_dir):
        for d in list(dirnames):
            if d.startswith("."):
                hidden_dir = os.path.join(current_root, d)
                try:
                    shutil.rmtree(hidden_dir)
                    removed += 1
                    print(f"Removed hidden folder: {hidden_dir}")
                except Exception:
                    pass
    if removed == 0:
        print("No hidden folders found.")


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def collect_class_images(root_dir: str):
    class_to_images = {}

    class_folders = [
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f)) and not f.startswith(".")
    ]
    class_folders.sort()

    for class_name in class_folders:
        class_dir = os.path.join(root_dir, class_name)
        image_paths = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(glob(os.path.join(class_dir, ext)))

        image_paths = sorted(image_paths)
        if len(image_paths) == 0:
            print(f"Warning: no supported images found in class [{class_name}]")
            continue

        class_to_images[class_name] = image_paths

    return class_to_images


def split_one_class(image_paths, train_ratio, val_ratio, test_ratio, seed):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    image_paths = image_paths.copy()
    rnd = random.Random(seed)
    rnd.shuffle(image_paths)

    n = len(image_paths)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    if n_train + n_val > n:
        n_val = n - n_train
    n_test = n - n_train - n_val

    train_paths = image_paths[:n_train]
    val_paths = image_paths[n_train:n_train + n_val]
    test_paths = image_paths[n_train + n_val:]

    return train_paths, val_paths, test_paths


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img

    if img.dtype == np.uint16:
        max_val = float(img.max()) if img.max() > 0 else 1.0
        img = (img.astype(np.float32) / max_val) * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


def read_rgb_image(path: str):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    if image.ndim == 2:
        image = normalize_to_uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    if image.ndim == 3:
        image = normalize_to_uint8(image)

        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            return image

        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        return None

    return None


def save_rgb_png(path: str, image_rgb: np.ndarray):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)


def copy_and_standardize_images(image_paths, dst_dir, prefix="orig"):
    ensure_dir(dst_dir)
    saved_paths = []

    for idx, src_path in enumerate(tqdm(image_paths, desc=f"Copy -> {os.path.basename(dst_dir)}", leave=False)):
        image = read_rgb_image(src_path)
        if image is None:
            print(f"Skipped unreadable image: {src_path}")
            continue

        base = os.path.splitext(os.path.basename(src_path))[0]
        save_name = f"{prefix}_{idx:06d}_{base}.png"
        save_path = os.path.join(dst_dir, save_name)

        save_rgb_png(save_path, image)
        saved_paths.append(save_path)

    return saved_paths


def compute_adaptive_target(train_count_dict):
    counts = sorted(train_count_dict.values(), reverse=True)

    if len(counts) == 1:
        return counts[0]

    # ignore the largest dominant class
    minority_counts = counts[1:]

    target = int(math.ceil(np.quantile(minority_counts, 0.75)))
    target = max(target, int(np.median(minority_counts)))

    return target


def get_target_count_for_class(original_train_count, global_target):
    if BALANCE_MODE == "none":
        return original_train_count

    if BALANCE_MODE == "fixed":
        raw_target = FIXED_TARGET_COUNT
    elif BALANCE_MODE == "adaptive":
        raw_target = global_target
    else:
        raise ValueError(f"Unsupported BALANCE_MODE: {BALANCE_MODE}")

    if original_train_count >= raw_target:
        return original_train_count

    capped_target = min(raw_target, original_train_count * MAX_AUG_RATIO)
    return max(original_train_count, capped_target)


def augment_train_class_to_target(class_name, original_train_paths, train_output_dir, target_count):
    current_count = len(original_train_paths)

    if target_count <= current_count:
        print(f"[{class_name}] train count = {current_count}, no augmentation needed.")
        return current_count, 0

    num_to_generate = target_count - current_count
    print(f"[{class_name}] augmenting train set: {current_count} -> {target_count} (+{num_to_generate})")

    aug_counter = 0
    pbar = tqdm(total=num_to_generate, desc=f"Augment -> {class_name}", leave=False)

    while aug_counter < num_to_generate:
        src_path = random.choice(original_train_paths)
        image = read_rgb_image(src_path)
        if image is None:
            continue

        aug_image = augmentor(image=image)["image"]

        src_base = os.path.splitext(os.path.basename(src_path))[0]
        save_name = f"aug_{aug_counter:06d}_from_{src_base}.png"
        save_path = os.path.join(train_output_dir, save_name)

        save_rgb_png(save_path, aug_image)

        aug_counter += 1
        pbar.update(1)

    pbar.close()
    return target_count, num_to_generate


def main():
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("Preparing standardized dataset with train-only augmentation")
    print("=" * 80)
    print(f"Albumentations version: {A.__version__}")

    remove_hidden_folders(INPUT_DATA_DIR)
    reset_output_dir(OUTPUT_ROOT_DIR)

    for split in ["train", "val", "test"]:
        ensure_dir(os.path.join(OUTPUT_ROOT_DIR, split))

    class_to_images = collect_class_images(INPUT_DATA_DIR)
    if len(class_to_images) == 0:
        raise RuntimeError("No valid image files were found.")

    original_distribution = {
        class_name: len(paths) for class_name, paths in class_to_images.items()
    }
    save_json(
        original_distribution,
        os.path.join(OUTPUT_ROOT_DIR, CLASS_DISTRIBUTION_JSON)
    )

    print("\nDetected class distribution:")
    total_images = 0
    for class_name, count in original_distribution.items():
        total_images += count
        print(f"  {class_name}: {count}")
    print(f"  Total: {total_images}")

    split_records = {}
    copied_train_originals = {}

    print("\n" + "=" * 80)
    print("Step 1: splitting raw dataset into train / val / test")
    print("=" * 80)

    for class_name, image_paths in class_to_images.items():
        print(f"\nProcessing class: [{class_name}]")
        print(f"Original total count: {len(image_paths)}")

        train_paths, val_paths, test_paths = split_one_class(
            image_paths=image_paths,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            seed=RANDOM_SEED
        )

        train_dir = os.path.join(OUTPUT_ROOT_DIR, "train", class_name)
        val_dir = os.path.join(OUTPUT_ROOT_DIR, "val", class_name)
        test_dir = os.path.join(OUTPUT_ROOT_DIR, "test", class_name)

        copied_train_paths = copy_and_standardize_images(train_paths, train_dir, prefix="orig")
        copied_val_paths = copy_and_standardize_images(val_paths, val_dir, prefix="orig")
        copied_test_paths = copy_and_standardize_images(test_paths, test_dir, prefix="orig")

        copied_train_originals[class_name] = copied_train_paths

        split_records[class_name] = {
            "original_total": len(image_paths),
            "train_original": len(copied_train_paths),
            "val_original": len(copied_val_paths),
            "test_original": len(copied_test_paths),
        }

        print(
            f"Split done -> train: {len(copied_train_paths)}, "
            f"val: {len(copied_val_paths)}, test: {len(copied_test_paths)}"
        )

    train_count_dict = {
        class_name: len(paths)
        for class_name, paths in copied_train_originals.items()
    }

    if BALANCE_MODE == "adaptive":
        global_target = compute_adaptive_target(train_count_dict)
    elif BALANCE_MODE == "fixed":
        global_target = FIXED_TARGET_COUNT
    else:
        global_target = None

    print("\n" + "=" * 80)
    print("Step 2: computing augmentation targets")
    print("=" * 80)
    print(f"BALANCE_MODE: {BALANCE_MODE}")
    if global_target is not None:
        print(f"Global target count for minority train classes: {global_target}")
    else:
        print("No offline augmentation target will be used.")

    print("\n" + "=" * 80)
    print("Step 3: augmenting training set only")
    print("=" * 80)

    for class_name, original_train_paths in copied_train_originals.items():
        train_dir = os.path.join(OUTPUT_ROOT_DIR, "train", class_name)
        original_train_count = len(original_train_paths)

        target_count = get_target_count_for_class(
            original_train_count=original_train_count,
            global_target=global_target
        )

        final_count, num_augmented = augment_train_class_to_target(
            class_name=class_name,
            original_train_paths=original_train_paths,
            train_output_dir=train_dir,
            target_count=target_count
        )

        split_records[class_name]["train_target"] = target_count
        split_records[class_name]["train_final"] = final_count
        split_records[class_name]["train_augmented"] = num_augmented

    summary_path = os.path.join(OUTPUT_ROOT_DIR, SUMMARY_JSON)
    save_json(split_records, summary_path)

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Prepared dataset saved to: {OUTPUT_ROOT_DIR}")
    print(f"Class distribution saved to: {os.path.join(OUTPUT_ROOT_DIR, CLASS_DISTRIBUTION_JSON)}")
    print(f"Split summary saved to: {summary_path}")
    print("\nFinal structure:")
    print(f"{OUTPUT_ROOT_DIR}/train/<class_name>")
    print(f"{OUTPUT_ROOT_DIR}/val/<class_name>")
    print(f"{OUTPUT_ROOT_DIR}/test/<class_name>")
    print("\nAll output images are standardized PNG files.")


if __name__ == "__main__":
    main()