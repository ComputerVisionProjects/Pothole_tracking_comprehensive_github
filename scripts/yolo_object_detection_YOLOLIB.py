import cv2
from ultralytics import YOLO
import torch
import time

# Set CUDA device
torch.cuda.set_device(0)

# Load the YOLOv8 model
model = YOLO(r"D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\models\yolov11n_best_25k.pt")

# Run detection on an image once (optional, just testing)
image_path = r"D:\Arybhatta_motors_computer_vision\Yolov8_custom\scripts\videos_images\istockphoto-174662203-612x612.jpg"
image_results = model(image_path)

# Video path
video_path = r"D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\Videos\VID_20250610_165403943.mp4"
cap = cv2.VideoCapture(video_path)

arr = []  # to store inference times

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success or frame is None:
        break

    # Resize frame if needed
    frame = cv2.resize(frame, (640, 640))
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # Run YOLOv8 detection
    start_time = time.time()
    results = model(frame, conf = 0.2)
    end_time = time.time()

    # FPS calculation
    inference_time = end_time - start_time
    arr.append(inference_time)
    FPS = str(int(1 / inference_time)) if inference_time > 0 else "0"

    # Plot results
    annotated_frame = results[0].plot()

    # Optional: Print box coordinates
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        print("Box coordinates:", x1.item(), y1.item(), x2.item(), y2.item())

    # Draw FPS on frame
    cv2.putText(annotated_frame, f"FPS = {FPS}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (178, 255, 102), 2)
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()

# Print average inference time
print("Average inference time per frame:", sum(arr) / len(arr))
