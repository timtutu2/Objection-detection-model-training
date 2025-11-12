import os
import random
import shutil
from tqdm import tqdm

SRC_DIR = "/home/tim/Desktop/ECE253/dataset/All/car_train/cars_train"
DST_DIR = "/home/tim/Desktop/ECE253/dataset/All/car_train_split"
train_ratio = 0.8  # 8:2 you can change the ratio if you want

# create output directories
train_img_dir = os.path.join(DST_DIR, "images/train")
val_img_dir   = os.path.join(DST_DIR, "images/val")
train_lbl_dir = os.path.join(DST_DIR, "labels/train")
val_lbl_dir   = os.path.join(DST_DIR, "labels/val")

for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# get all images
images = [f for f in os.listdir(SRC_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(images)
split_idx = int(len(images) * train_ratio)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

# assume labels and images have the same name and exist in another directory (e.g. cars_train_label)
label_dir = SRC_DIR.replace("cars_train", "car_train_label")  # if different name, change this line

def copy_files(img_list, dst_img_dir, dst_lbl_dir):
    for img in tqdm(img_list):
        name, _ = os.path.splitext(img)
        src_img = os.path.join(SRC_DIR, img)
        src_lbl = os.path.join(label_dir, name + ".txt")

        shutil.copy(src_img, dst_img_dir)
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl_dir)

print(f"Total images: {len(images)}")
print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")

print("\nCopying train set...")
copy_files(train_imgs, train_img_dir, train_lbl_dir)

print("\nCopying val set...")
copy_files(val_imgs, val_img_dir, val_lbl_dir)

print("\n Done! Split 8:2 completed.")
