import cv2
import numpy as np

# === CONFIGURATION ===
input_video_path = r"D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\Videos\output_transformed.avi"
output_video_path = r"D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\Videos\output_videos.mp4"

# Source points (from the original perspective)
src_pts = np.float32([[148, 275],[408, 274],[28, 359], [549, 333]       
])

input_size = (1280, 720)

# Destination points (top-down view)
dst_width = 640
dst_height = 480
dst_pts = np.float32([
    [0, 0],
    [dst_width, 0],
    [0, dst_height],
    [dst_width, dst_height]
])

# === OPEN VIDEO ===
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {input_video_path}")

# Get frame properties
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec if needed

# Output video writer
out = cv2.VideoWriter(output_video_path, fourcc, fps, (dst_width, dst_height))

# Compute perspective matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# === FRAME LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply perspective warp
    resized_img = cv2.resize(frame, input_size)
    resized_img = cv2.flip(resized_img, 0)

    warped = cv2.warpPerspective(resized_img, M, (dst_width, dst_height))

    # Write to output video
    out.write(warped)

# === CLEANUP ===
cap.release()
out.release()
print(f"Saved warped video to {output_video_path}")
