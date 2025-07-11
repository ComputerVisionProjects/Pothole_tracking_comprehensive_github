import cv2
import os
import glob
import numpy as np

# === CONFIGURATION ===
input_dir = "D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\pptrnsfrm_images_lab"
output_dir = "D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\pptrnsfrm_saved_after"
os.makedirs(output_dir, exist_ok=True)

# Resize input images to 1280x720 (720p)
input_size = (1280, 720)

# Destination size: 854x480 (480p)
output_width = 640
output_height = 480

# Define source and destination points
# These source points are for 1280x720 input images
src_pts = np.float32([
    [240, 337],   # top-left
    [1037, 337],   # top-right
    [19, 530],   # bottom-left
    [1225, 520]  
])

dst_pts = np.float32([
    [0, 0],
    [output_width, 0],
    [0, output_height],
    [output_width, output_height]
])

# Compute the perspective transform matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# === MAIN LOOP ===
image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + \
              glob.glob(os.path.join(input_dir, "*.png")) + \
              glob.glob(os.path.join(input_dir, "*.jpeg"))

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}, skipping.")
        continue

    # Resize to 1280x720 before transformation
    resized_img = cv2.resize(img, input_size)

    # Apply perspective warp to get 854x480 output
    warped = cv2.warpPerspective(resized_img, M, (output_width, output_height))

    # Save transformed image
    filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, warped)
    print(f"Saved: {output_path}")
