import os

# Path to the directory containing YOLO label `.txt` files
label_dir = r"Pakistani License Plates.v2i.yolov11\valid\labels"

# Set of class IDs you want to keep (e.g., {3})
classes_to_keep = {1}

for filename in os.listdir(label_dir):
    if not filename.endswith(".txt"):
        continue

    path = os.path.join(label_dir, filename)

    with open(path, "r") as f:
        lines = f.readlines()

    filtered_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        class_id = int(parts[0])
        if class_id in classes_to_keep:
            # Replace class ID with 0
            parts[0] = "0"
            filtered_lines.append(" ".join(parts) + "\n")

    with open(path, "w") as f:
        f.writelines(filtered_lines)
