import cv2
import os

# Folder containing your images
folder_path = "D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\pptrnsfrm_images_lab"

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# Flip code: 0 = vertical, 1 = horizontal, -1 = both
flip_code = 0  # Change if needed

# Iterate through all files in the folder
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(image_extensions):
            image_path = os.path.join(root, file)

            # Read the image
            image = cv2.imread(image_path)

            if image is not None:
                # Flip the image
                flipped = cv2.flip(image, flip_code)

                # Overwrite the original image
                cv2.imwrite(image_path, flipped)
                print(f"Flipped and overwritten: {image_path}")
            else:
                print(f"Failed to read image: {image_path}")
