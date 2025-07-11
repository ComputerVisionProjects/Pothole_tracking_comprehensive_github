import cv2

from ultralytics import YOLO

import torch

# torch.cuda.set_device(0)

# Load the YOLOv8 model
model = YOLO(r"E:\Aryabhatta_motors_computer_vision\scripts\models\best.onnx")
# print("before: ",model.device.type)
# results = model(r"E:\Aryabhatta_motors_computer_vision\images_potholes_1\dataset-pothole\yolov8_custom\train\images\01_jpg.rf.3ca97922642224c05e3602b324e899f2.jpg")
# Open the video file
# print("after: ",model.device.type)
video_path = r"E:\Aryabhatta_motors_computer_vision\scripts\videos\WhatsApp Video 2024-07-05 at 01.41.50_72d4a5c5 (online-video-cutter.com).mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

   
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates in format [x1, y1, x2, y2]
                print("Box coordinates:", x1, y1, x2, y2)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        print(annotated_frame)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()