import os
import shutil
import glob

# === CONFIGURATION ===
input_dir = "D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\pptrnsfrm_saved_after"
output_dir = "D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\pptrnsfrm_images_train_yolo"
os.makedirs(output_dir, exist_ok=True)

# === MAIN LOGIC ===
image_extensions = ('*.jpg', '*.jpeg', '*.png')
image_files = []

for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(input_dir, ext)))

for img_path in image_files:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(input_dir, base_name + ".txt")

    if os.path.exists(label_path):
        # Copy image
        img_dest = os.path.join(output_dir, os.path.basename(img_path))
        shutil.copyfile(img_path, img_dest)

        # Copy label
        label_dest = os.path.join(output_dir, os.path.basename(label_path))
        shutil.copyfile(label_path, label_dest)

        print(f"Copied: {img_dest} and {label_dest}")
