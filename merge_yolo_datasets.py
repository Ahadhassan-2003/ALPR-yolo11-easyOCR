import os
import shutil
from pathlib import Path

# List of dataset root directories
datasets = [
    "Car Number Plate Detection.v1i.yolov11",
    "model.v1i.yolov11",
    "numberplate.v1i.yolov11",
    "Pakistani License Plates.v2i.yolov11",
    "Pakistani-Number-plates.v1i.yolov11"
]

# Destination merged dataset
merged_root = Path("Pakistani License Plates (Merged) - YOLOv11")

# Create all required merged folders
subsets = ["train", "valid", "test"]
for subset in subsets:
    (merged_root / "images" / subset).mkdir(parents=True, exist_ok=True)
    (merged_root / "labels" / subset).mkdir(parents=True, exist_ok=True)

def copy_and_rename(src_dir, dst_dir, prefix):
    if not src_dir.exists():
        return
    for file in src_dir.glob("*.*"):
        if file.is_file():
            new_name = f"{prefix}_{file.name}"
            shutil.copy2(file, dst_dir / new_name)

# Loop through datasets
for idx, dataset in enumerate(datasets):
    prefix = f"ds{idx+1}"
    dataset_path = Path(dataset)

    for subset in subsets:
        for actual_name in [subset, "valid" if subset == "val" else subset]:
            subset_path = dataset_path / actual_name
            if not subset_path.exists():
                continue

            img_dir = subset_path / "images"
            lbl_dir = subset_path / "labels"

            merged_img_dir = merged_root / "images" / subset
            merged_lbl_dir = merged_root / "labels" / subset

            print(f"\n📂 Merging {prefix} → {subset}")
            print(f"  Images: {img_dir} → {merged_img_dir}")
            print(f"  Labels: {lbl_dir} → {merged_lbl_dir}")

            copy_and_rename(img_dir, merged_img_dir, prefix)
            copy_and_rename(lbl_dir, merged_lbl_dir, prefix)

print("\n✅ Merge completed")
