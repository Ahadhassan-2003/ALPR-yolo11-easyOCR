import os
from pathlib import Path

# Set your dataset root directory
dataset_dir = Path("Pakistani License Plates (Merged) - YOLOv11")  # or your dataset folder

# Folder names (adjust if needed)
splits = {
    "train": dataset_dir / "images" / "train",
    "valid": dataset_dir / "images" / "valid",
    "test": dataset_dir / "images" / "test"
}

# Supported image extensions
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# First count images per split
split_counts = {}
total = 0

for split_name, folder in splits.items():
    if folder.exists():
        count = sum(1 for f in folder.glob("*") if f.suffix.lower() in image_exts)
        split_counts[split_name] = count
        total += count
    else:
        split_counts[split_name] = None  # Folder missing

# Print results with percentages
print("📊 Split Summary:\n")
for split_name, count in split_counts.items():
    if count is not None:
        percent = (count / total * 100) if total > 0 else 0
        print(f"{split_name.capitalize():<6} → {count:>4} images ({percent:5.1f}%)")
    else:
        print(f"{split_name.capitalize():<6} → Folder not found")

print(f"\n🧮 Total images: {total}")
