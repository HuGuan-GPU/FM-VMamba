from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
import numpy as np


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.astype(np.uint8).ravel():
        value = (value << 1) | int(bit)
    return value


def _hash_int_to_hex(value: int, bit_count: int) -> str:
    width = max(1, (bit_count + 3) // 4)
    return f"{value:0{width}x}"


@lru_cache(maxsize=8)
def _dct_matrix(size: int) -> np.ndarray:
    n = np.arange(size, dtype=np.float64)
    k = n[:, None]
    matrix = np.cos((np.pi / size) * (n + 0.5) * k)
    matrix[0, :] *= np.sqrt(1.0 / size)
    matrix[1:, :] *= np.sqrt(2.0 / size)
    return matrix


def compute_phash(image: Image.Image, hash_size: int) -> tuple[int, str]:
    sample_size = hash_size * 4
    gray = image.convert("L").resize(
        (sample_size, sample_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(gray, dtype=np.float64)
    dct_basis = _dct_matrix(sample_size)
    dct_values = dct_basis @ pixels @ dct_basis.T
    low_frequency = dct_values[:hash_size, :hash_size]
    median = np.median(low_frequency.ravel()[1:])
    bits = low_frequency > median
    value = _bits_to_int(bits)
    return value, _hash_int_to_hex(value, hash_size * hash_size)


def compute_dhash(image: Image.Image, hash_size: int) -> tuple[int, str]:
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = _bits_to_int(bits)
    return value, _hash_int_to_hex(value, hash_size * hash_size)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    class_name: str
    relative_path: str
    absolute_path: str
    file_size_bytes: int
    width: int
    height: int
    mode: str
    sha256: str
    phash_hex: str
    dhash_hex: str
    phash_int: int
    dhash_int: int


@dataclass(frozen=True)
class InvalidImageRecord:
    relative_path: str
    absolute_path: str
    error_type: str
    error_message: str


@dataclass(frozen=True)
class CandidatePair:
    image_id_1: str
    image_id_2: str
    class_1: str
    class_2: str
    relative_path_1: str
    relative_path_2: str
    sha256_equal: bool
    phash_distance: int
    dhash_distance: int
    cross_class: bool
    severity: str


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        item_list = list(items)
        self.parent = {item: item for item in item_list}
        self.rank = {item: 0 for item in item_list}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, item_a: str, item_b: str) -> None:
        root_a = self.find(item_a)
        root_b = self.find(item_b)
        if root_a == root_b:
            return
        rank_a = self.rank[root_a]
        rank_b = self.rank[root_b]
        if rank_a < rank_b:
            self.parent[root_a] = root_b
        elif rank_a > rank_b:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen a class-folder image dataset for exact and near-duplicate "
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root. Each class should be stored in a subdirectory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/duplicate_audit"),
        help="Directory used to store audit reports.",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=4,
        help="Maximum pHash Hamming distance for a candidate pair (default: 4).",
    )
    parser.add_argument(
        "--dhash-threshold",
        type=int,
        default=6,
        help="Maximum dHash Hamming distance for a candidate pair (default: 6).",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=8,
        help="Perceptual hash size. 8 means 64-bit hashes (default: 8).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of parallel image-reading workers (default: up to 8).",
    )
    parser.add_argument(
        "--make-contact-sheets",
        action="store_true",
        help="Create side-by-side JPEG previews for candidate pairs.",
    )
    parser.add_argument(
        "--max-contact-sheets",
        type=int,
        default=200,
        help="Maximum number of contact sheets to create (default: 200).",
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=360,
        help="Long-edge size for contact-sheet thumbnails (default: 360).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def discover_images(data_root: Path) -> list[Path]:
    paths = [
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.as_posix().lower())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def class_name_from_path(path: Path, data_root: Path) -> str:
    relative = path.relative_to(data_root)
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return path.parent.name


def process_one_image(
    index: int,
    path: Path,
    data_root: Path,
    hash_size: int,
) -> ImageRecord | InvalidImageRecord:
    relative = path.relative_to(data_root).as_posix()
    image_id = f"IMG{index:06d}"
    try:
        file_size = path.stat().st_size
        file_digest = sha256_file(path)
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
            original_mode = image.mode
            rgb = image.convert("RGB")
            phash_int, phash_hex = compute_phash(rgb, hash_size=hash_size)
            dhash_int, dhash_hex = compute_dhash(rgb, hash_size=hash_size)

        return ImageRecord(
            image_id=image_id,
            class_name=class_name_from_path(path, data_root),
            relative_path=relative,
            absolute_path=str(path.resolve()),
            file_size_bytes=file_size,
            width=width,
            height=height,
            mode=original_mode,
            sha256=file_digest,
            phash_hex=phash_hex,
            dhash_hex=dhash_hex,
            phash_int=phash_int,
            dhash_int=dhash_int,
        )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return InvalidImageRecord(
            relative_path=relative,
            absolute_path=str(path.resolve()),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def process_images(
    paths: Sequence[Path],
    data_root: Path,
    hash_size: int,
    workers: int,
) -> tuple[list[ImageRecord], list[InvalidImageRecord]]:
    records: list[ImageRecord] = []
    invalid: list[InvalidImageRecord] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(process_one_image, idx, path, data_root, hash_size): path
            for idx, path in enumerate(paths, start=1)
        }
        for completed, future in enumerate(as_completed(future_map), start=1):
            result = future.result()
            if isinstance(result, ImageRecord):
                records.append(result)
            else:
                invalid.append(result)
            if completed % 100 == 0 or completed == len(paths):
                logging.info("Hashed %d/%d images", completed, len(paths))

    records.sort(key=lambda item: item.image_id)
    invalid.sort(key=lambda item: item.relative_path)
    return records, invalid


def hamming_distance(value_a: int, value_b: int) -> int:
    return (value_a ^ value_b).bit_count()


def pair_severity(
    sha256_equal: bool,
    phash_distance: int,
    dhash_distance: int,
) -> str:
    if sha256_equal:
        return "exact"
    if phash_distance <= 2 and dhash_distance <= 3:
        return "strong"
    if phash_distance <= 4 and dhash_distance <= 6:
        return "probable"
    return "review"


def build_candidate_pairs(
    records: Sequence[ImageRecord],
    phash_threshold: int,
    dhash_threshold: int,
) -> list[CandidatePair]:
    candidates: list[CandidatePair] = []
    total_pairs = len(records) * (len(records) - 1) // 2
    logging.info("Comparing %d image pairs", total_pairs)

    for record_a, record_b in itertools.combinations(records, 2):
        same_sha = record_a.sha256 == record_b.sha256
        phash_dist = hamming_distance(record_a.phash_int, record_b.phash_int)

        if not same_sha and phash_dist > phash_threshold:
            continue

        dhash_dist = hamming_distance(record_a.dhash_int, record_b.dhash_int)
        if not same_sha and dhash_dist > dhash_threshold:
            continue

        candidates.append(
            CandidatePair(
                image_id_1=record_a.image_id,
                image_id_2=record_b.image_id,
                class_1=record_a.class_name,
                class_2=record_b.class_name,
                relative_path_1=record_a.relative_path,
                relative_path_2=record_b.relative_path,
                sha256_equal=same_sha,
                phash_distance=phash_dist,
                dhash_distance=dhash_dist,
                cross_class=record_a.class_name != record_b.class_name,
                severity=pair_severity(same_sha, phash_dist, dhash_dist),
            )
        )

    severity_order = {"exact": 0, "strong": 1, "probable": 2, "review": 3}
    candidates.sort(
        key=lambda pair: (
            0 if pair.cross_class else 1,
            severity_order[pair.severity],
            pair.phash_distance,
            pair.dhash_distance,
            pair.relative_path_1,
            pair.relative_path_2,
        )
    )
    return candidates


def build_exact_duplicate_groups(
    records: Sequence[ImageRecord],
) -> list[list[ImageRecord]]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.sha256].append(record)
    duplicate_groups = [items for items in groups.values() if len(items) > 1]
    duplicate_groups.sort(
        key=lambda group: (-len(group), [item.relative_path for item in group])
    )
    return duplicate_groups


def build_candidate_clusters(
    records: Sequence[ImageRecord],
    candidates: Sequence[CandidatePair],
) -> list[list[str]]:
    union_find = UnionFind(record.image_id for record in records)
    involved_ids: set[str] = set()

    for pair in candidates:
        union_find.union(pair.image_id_1, pair.image_id_2)
        involved_ids.add(pair.image_id_1)
        involved_ids.add(pair.image_id_2)

    clusters_by_root: dict[str, list[str]] = defaultdict(list)
    for image_id in sorted(involved_ids):
        clusters_by_root[union_find.find(image_id)].append(image_id)

    clusters = [sorted(cluster) for cluster in clusters_by_root.values() if len(cluster) > 1]
    clusters.sort(key=lambda cluster: (-len(cluster), cluster))
    return clusters


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_inventory(output_dir: Path, records: Sequence[ImageRecord]) -> None:
    rows = []
    for record in records:
        row = asdict(record)
        row.pop("phash_int")
        row.pop("dhash_int")
        rows.append(row)
    fieldnames = [
        "image_id",
        "class_name",
        "relative_path",
        "absolute_path",
        "file_size_bytes",
        "width",
        "height",
        "mode",
        "sha256",
        "phash_hex",
        "dhash_hex",
    ]
    write_csv(output_dir / "image_inventory.csv", rows, fieldnames)


def write_invalid_images(
    output_dir: Path,
    invalid: Sequence[InvalidImageRecord],
) -> None:
    rows = [asdict(item) for item in invalid]
    fieldnames = ["relative_path", "absolute_path", "error_type", "error_message"]
    write_csv(output_dir / "invalid_images.csv", rows, fieldnames)


def write_exact_duplicate_pairs(
    output_dir: Path,
    duplicate_groups: Sequence[Sequence[ImageRecord]],
) -> int:
    rows: list[dict] = []
    pair_count = 0
    for group_index, group in enumerate(duplicate_groups, start=1):
        group_id = f"EXACT{group_index:04d}"
        for record_a, record_b in itertools.combinations(group, 2):
            pair_count += 1
            rows.append(
                {
                    "exact_group_id": group_id,
                    "image_id_1": record_a.image_id,
                    "image_id_2": record_b.image_id,
                    "class_1": record_a.class_name,
                    "class_2": record_b.class_name,
                    "relative_path_1": record_a.relative_path,
                    "relative_path_2": record_b.relative_path,
                    "cross_class": record_a.class_name != record_b.class_name,
                    "sha256": record_a.sha256,
                }
            )
    fieldnames = [
        "exact_group_id",
        "image_id_1",
        "image_id_2",
        "class_1",
        "class_2",
        "relative_path_1",
        "relative_path_2",
        "cross_class",
        "sha256",
    ]
    write_csv(output_dir / "exact_duplicate_pairs.csv", rows, fieldnames)
    return pair_count


def write_near_duplicate_candidates(
    output_dir: Path,
    candidates: Sequence[CandidatePair],
) -> None:
    rows = []
    for index, pair in enumerate(candidates, start=1):
        row = asdict(pair)
        row = {
            "candidate_id": f"PAIR{index:06d}",
            **row,
            "manual_decision": "",
            "manual_notes": "",
        }
        rows.append(row)
    fieldnames = [
        "candidate_id",
        "image_id_1",
        "image_id_2",
        "class_1",
        "class_2",
        "relative_path_1",
        "relative_path_2",
        "sha256_equal",
        "phash_distance",
        "dhash_distance",
        "cross_class",
        "severity",
        "manual_decision",
        "manual_notes",
    ]
    write_csv(output_dir / "near_duplicate_candidates.csv", rows, fieldnames)


def write_candidate_clusters(
    output_dir: Path,
    clusters: Sequence[Sequence[str]],
    records_by_id: dict[str, ImageRecord],
) -> None:
    rows: list[dict] = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        cluster_id = f"CLUSTER{cluster_index:04d}"
        classes = sorted({records_by_id[image_id].class_name for image_id in cluster})
        class_text = "|".join(classes)
        for image_id in cluster:
            record = records_by_id[image_id]
            rows.append(
                {
                    "candidate_cluster_id": cluster_id,
                    "cluster_size": len(cluster),
                    "classes_in_cluster": class_text,
                    "cross_class_cluster": len(classes) > 1,
                    "image_id": record.image_id,
                    "class_name": record.class_name,
                    "relative_path": record.relative_path,
                    "manual_cluster_decision": "",
                    "manual_notes": "",
                }
            )
    fieldnames = [
        "candidate_cluster_id",
        "cluster_size",
        "classes_in_cluster",
        "cross_class_cluster",
        "image_id",
        "class_name",
        "relative_path",
        "manual_cluster_decision",
        "manual_notes",
    ]
    write_csv(output_dir / "candidate_clusters.csv", rows, fieldnames)


def load_thumbnail(path: Path, target_size: int) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_size, target_size), "white")
        offset_x = (target_size - image.width) // 2
        offset_y = (target_size - image.height) // 2
        canvas.paste(image, (offset_x, offset_y))
        return canvas


def create_contact_sheets(
    output_dir: Path,
    candidates: Sequence[CandidatePair],
    records_by_id: dict[str, ImageRecord],
    max_sheets: int,
    thumbnail_size: int,
) -> int:
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    created = 0

    for index, pair in enumerate(candidates[:max_sheets], start=1):
        record_a = records_by_id[pair.image_id_1]
        record_b = records_by_id[pair.image_id_2]
        try:
            image_a = load_thumbnail(Path(record_a.absolute_path), thumbnail_size)
            image_b = load_thumbnail(Path(record_b.absolute_path), thumbnail_size)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logging.warning("Could not create contact sheet for pair %d: %s", index, exc)
            continue

        header_height = 72
        margin = 12
        width = thumbnail_size * 2 + margin * 3
        height = thumbnail_size + header_height + margin * 2
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)

        draw.text(
            (margin, 8),
            (
                f"PAIR{index:06d} | severity={pair.severity} | "
                f"pHash={pair.phash_distance} | dHash={pair.dhash_distance} | "
                f"cross_class={pair.cross_class}"
            ),
            fill="black",
            font=font,
        )
        draw.text(
            (margin, 30),
            f"A: {record_a.image_id} (see CSV for path/class)",
            fill="black",
            font=font,
        )
        draw.text(
            (margin + thumbnail_size + margin, 30),
            f"B: {record_b.image_id} (see CSV for path/class)",
            fill="black",
            font=font,
        )

        top = header_height + margin
        sheet.paste(image_a, (margin, top))
        sheet.paste(image_b, (thumbnail_size + margin * 2, top))
        sheet.save(contact_dir / f"PAIR{index:06d}.jpg", quality=92)
        created += 1

    return created


def write_summary(
    output_dir: Path,
    data_root: Path,
    records: Sequence[ImageRecord],
    invalid: Sequence[InvalidImageRecord],
    duplicate_groups: Sequence[Sequence[ImageRecord]],
    exact_pair_count: int,
    candidates: Sequence[CandidatePair],
    clusters: Sequence[Sequence[str]],
    phash_threshold: int,
    dhash_threshold: int,
    hash_size: int,
    contact_sheet_count: int,
    elapsed_seconds: float,
) -> dict:
    class_counts = Counter(record.class_name for record in records)
    severity_counts = Counter(pair.severity for pair in candidates)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "total_discovered_images": len(records) + len(invalid),
        "valid_images": len(records),
        "invalid_images": len(invalid),
        "class_counts": dict(sorted(class_counts.items())),
        "hash_configuration": {
            "hash_size": hash_size,
            "hash_bits": hash_size * hash_size,
            "phash_threshold": phash_threshold,
            "dhash_threshold": dhash_threshold,
            "candidate_rule": (
                "sha256_equal OR "
                "(phash_distance <= phash_threshold AND "
                "dhash_distance <= dhash_threshold)"
            ),
        },
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_pairs": exact_pair_count,
        "near_duplicate_candidate_pairs": len(candidates),
        "cross_class_candidate_pairs": sum(pair.cross_class for pair in candidates),
        "candidate_pair_severity_counts": dict(sorted(severity_counts.items())),
        "candidate_clusters": len(clusters),
        "contact_sheets_created": contact_sheet_count,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "interpretation_note": (
            "Near-duplicate results are screening candidates and require manual review. "
            "The script does not modify source images. Screening alone does not remove "
            "cross-validation leakage. Confirmed same-scene or same-examination images "
            "should be assigned to the same fold and affected experiments rerun."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not data_root.exists() or not data_root.is_dir():
        logging.error("Dataset root does not exist or is not a directory: %s", data_root)
        return 2
    if args.phash_threshold < 0 or args.dhash_threshold < 0:
        logging.error("Hash thresholds must be non-negative integers.")
        return 2
    if args.hash_size < 4:
        logging.error("hash-size must be at least 4.")
        return 2
    if args.workers < 1:
        logging.error("workers must be at least 1.")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    image_paths = discover_images(data_root)
    if not image_paths:
        logging.error("No supported image files were found under: %s", data_root)
        return 3

    logging.info("Found %d image files under %s", len(image_paths), data_root)
    records, invalid = process_images(
        image_paths,
        data_root=data_root,
        hash_size=args.hash_size,
        workers=args.workers,
    )

    write_inventory(output_dir, records)
    write_invalid_images(output_dir, invalid)

    duplicate_groups = build_exact_duplicate_groups(records)
    exact_pair_count = write_exact_duplicate_pairs(output_dir, duplicate_groups)

    candidates = build_candidate_pairs(
        records,
        phash_threshold=args.phash_threshold,
        dhash_threshold=args.dhash_threshold,
    )
    write_near_duplicate_candidates(output_dir, candidates)

    records_by_id = {record.image_id: record for record in records}
    clusters = build_candidate_clusters(records, candidates)
    write_candidate_clusters(output_dir, clusters, records_by_id)

    contact_sheet_count = 0
    if args.make_contact_sheets and candidates:
        contact_sheet_count = create_contact_sheets(
            output_dir,
            candidates,
            records_by_id,
            max_sheets=args.max_contact_sheets,
            thumbnail_size=args.thumbnail_size,
        )

    elapsed = time.time() - start_time
    summary = write_summary(
        output_dir=output_dir,
        data_root=data_root,
        records=records,
        invalid=invalid,
        duplicate_groups=duplicate_groups,
        exact_pair_count=exact_pair_count,
        candidates=candidates,
        clusters=clusters,
        phash_threshold=args.phash_threshold,
        dhash_threshold=args.dhash_threshold,
        hash_size=args.hash_size,
        contact_sheet_count=contact_sheet_count,
        elapsed_seconds=elapsed,
    )

    logging.info("Audit completed. Reports saved to: %s", output_dir)
    logging.info(
        "Valid=%d | Invalid=%d | Exact groups=%d | Candidate pairs=%d | Cross-class=%d",
        summary["valid_images"],
        summary["invalid_images"],
        summary["exact_duplicate_groups"],
        summary["near_duplicate_candidate_pairs"],
        summary["cross_class_candidate_pairs"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
