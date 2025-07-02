import os

def count_files(root_dir):
    total_files = 0
    for _, _, files in os.walk(root_dir):
        total_files += len(files)
    return total_files

path = "Pakistani License Plates (Merged) - YOLOv11"
print(f"📦 Total files in '{path}':", count_files(path))