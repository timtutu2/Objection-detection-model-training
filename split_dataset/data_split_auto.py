import os
import random
import shutil
from pathlib import Path

# Change this to your dataset root folder
dataset_dir = Path("datasets/car_house")
image_dir = dataset_dir / "images"
label_dir = dataset_dir / "labels"

# Create output folders if not exist
for split in ["train", "val"]:
    os.makedirs(image_dir / split, exist_ok=True)
    os.makedirs(label_dir / split, exist_ok=True)

# Get all image files
all_images = list((image_dir).glob("*.jpg")) + list((image_dir).glob("*.png"))
random.shuffle(all_images)

# Split 90% train / 10% val
split_ratio = 0.9
split_idx = int(len(all_images) * split_ratio)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

def move_files(img_list, split):
    for img_path in img_list:
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        shutil.move(str(img_path), str(image_dir / split / img_path.name))
        shutil.move(str(label_path), str(label_dir / split / label_path.name))

move_files(train_images, "train")
move_files(val_images, "val")

print(f"Done! Train: {len(train_images)}  |  Val: {len(val_images)}")
