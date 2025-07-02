import os
from pathlib import Path

# === CONFIGURATION ===
dataset_dir = Path("Pakistani License Plates (Merged) - YOLOv11")
image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
nc = 1  # number of classes in your dataset
auto_confirm = False  # set to True to skip prompt

# === VALIDATION ===

def check_line_validity(line, nc):
    parts = line.strip().split()
    if len(parts) != 5:
        return False, f"Expected 5 elements, found {len(parts)}"
    try:
        class_id = int(parts[0])
    except ValueError:
        return False, f"Class ID '{parts[0]}' is not an integer"
    if not (0 <= class_id < nc):
        return False, f"Class ID '{class_id}' out of range [0, {nc - 1}]"
    try:
        coords = list(map(float, parts[1:]))
    except ValueError:
        return False, "One or more coordinates are not numeric"
    if not all(0.0 <= v <= 1.0 for v in coords):
        return False, f"Coordinates not in [0, 1]: {coords}"
    return True, ""

# === MAIN SCRIPT ===

invalid_files = []

for split in ["train", "valid", "test"]:
    label_dir = dataset_dir / "labels" / split
    image_dir = dataset_dir / "images" / split

    if not label_dir.exists():
        continue

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        reasons = []
        if not lines:
            reasons.append("Label file is empty")
        else:
            for i, line in enumerate(lines):
                valid, reason = check_line_validity(line, nc)
                if not valid:
                    reasons.append(f"Line {i+1}: {reason}")

        if reasons:
            # Try to find the corresponding image
            image_stem = label_file.stem
            img_path = None
            for ext in image_exts:
                candidate = image_dir / f"{image_stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            invalid_files.append({
                "label": label_file,
                "image": img_path,
                "reasons": reasons
            })

# === SUMMARY ===

print(f"\n📊 Validation Summary")
print(f"🔍 Total invalid files found: {len(invalid_files)}\n")

for entry in invalid_files:
    print(f"❌ {entry['label'].relative_to(dataset_dir)}")
    for r in entry['reasons']:
        print(f"   - {r}")
    if entry["image"]:
        print(f"   → Image: {entry['image'].relative_to(dataset_dir)}")
    else:
        print(f"   → Image not found.")

# === DELETION PROMPT ===

if invalid_files:
    if auto_confirm or input("\n🔥 Delete all these label/image pairs? [y/N]: ").lower() in {"y", "yes"}:
        for entry in invalid_files:
            try:
                os.remove(entry["label"])
                print(f"✅ Deleted label: {entry['label'].name}")
                if entry["image"] and entry["image"].exists():
                    os.remove(entry["image"])
                    print(f"✅ Deleted image: {entry['image'].name}")
            except Exception as e:
                print(f"❌ Error deleting: {e}")
    else:
        print("\n❎ Deletion cancelled.")
else:
    print("✅ All label files are valid!")
