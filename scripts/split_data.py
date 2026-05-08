import os, shutil, random

src_images = "dataset/images"
src_masks = "dataset/masks"

val_ratio = 0.2 


os.makedirs("dataset/val_images", exist_ok=True)
os.makedirs("dataset/val_masks", exist_ok=True)


files = os.listdir(src_images)
random.shuffle(files)

val_count = int(len(files) * val_ratio)
val_files = files[:val_count]


for f in val_files:
    shutil.move(os.path.join(src_images, f), "dataset/val_images/" + f)
    shutil.move(os.path.join(src_masks, f), "dataset/val_masks/" + f)

print(f"✅ Moved {val_count} files to validation set.")
