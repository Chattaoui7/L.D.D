import os
import shutil
import random

source_dir = r"C:\Users\admin\Desktop\Leaf-Disease-Detection\data\sunflower\training"
dest_dir = r"C:\Users\admin\Desktop\Leaf-Disease-Detection\data\sunflower\testing"

folders = os.listdir(source_dir)

for f in folders:
    source_folder = os.path.join(source_dir, f)
    dest_folder = os.path.join(dest_dir, f)

    # Create destination folder if it does not exist
    os.makedirs(dest_folder, exist_ok=True)

    files = os.listdir(source_folder)
    print(f"Processing folder '{f}' with {len(files)} files")

    # Choose number of samples to move, max 4 or all files if fewer
    n_samples = min(4, len(files))
    sample_indices = random.sample(range(len(files)), n_samples)
    print(f"Moving {n_samples} files: indices {sample_indices}")

    for idx in sample_indices:
        src_path = os.path.join(source_folder, files[idx])
        dst_path = os.path.join(dest_folder, files[idx])
        shutil.move(src_path, dst_path)
