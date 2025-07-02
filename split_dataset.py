import os
import shutil
from pathlib import Path
from math import ceil

# === CONFIGURATION ===

dataset_dir = Path("Pakistani License Plates (Merged) - YOLOv11")
image_exts = {".jpg", ".jpeg", ".png"}
target_split = {
    "train": 0.7,
    "valid": 0.2,
    "test": 0.1
}

source_priority = ["train", "valid", "test"]  # Used when rebalancing

# === END CONFIGURATION ===


def get_images(folder):
    return sorted([
        f for f in folder.glob("*") if f.suffix.lower() in image_exts and f.is_file()
    ])


def ensure_dirs():
    for s in ["train", "valid", "test"]:
        for t in ["images", "labels"]:
            (dataset_dir / t / s).mkdir(parents=True, exist_ok=True)


def move_pair(img_path, src_split, dst_split):
    label_path = dataset_dir / "labels" / src_split / img_path.with_suffix(".txt").name
    dst_img_path = dataset_dir / "images" / dst_split / img_path.name
    dst_lbl_path = dataset_dir / "labels" / dst_split / label_path.name

    shutil.move(str(img_path), dst_img_path)
    if label_path.exists():
        shutil.move(str(label_path), dst_lbl_path)


def compute_target_counts(total):
    result = {}
    remaining = total
    for split, frac in target_split.items():
        result[split] = ceil(total * frac)
        remaining -= result[split]
    # Adjust for rounding error
    last = list(target_split)[-1]
    result[last] += remaining
    return result


def count_current():
    return {
        split: len(get_images(dataset_dir / "images" / split))
        for split in ["train", "valid", "test"]
    }


def rebalance(current, target):
    for split in ["train", "valid", "test"]:
        need = target[split] - current[split]
        if need <= 0:
            continue

        print(f"🔄 Rebalancing: need {need} more in '{split}'")

        for src in source_priority:
            if src == split or current[src] <= target[src]:
                continue

            available = get_images(dataset_dir / "images" / src)
            to_move = min(len(available), need)
            for i in range(to_move):
                move_pair(available[i], src, split)
            current = count_current()  # Update counts
            need -= to_move
            if need <= 0:
                break


# --- MAIN EXECUTION ---

ensure_dirs()

# Count total images across all splits
total = sum(len(get_images(dataset_dir / "images" / s)) for s in ["train", "valid", "test"])
print(f"\n📦 Total images: {total}")

target_counts = compute_target_counts(total)
current_counts = count_current()

print(f"🎯 Target split: {target_counts}")
print(f"📊 Current split: {current_counts}")

rebalance(current_counts, target_counts)

# Final counts
final_counts = count_current()
print("\n✅ Final split counts:")
for split in ["train", "valid", "test"]:
    percent = final_counts[split] / total * 100 if total else 0
    print(f"  {split:<6} → {final_counts[split]:>4} images ({percent:.1f}%)")
