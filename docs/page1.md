# Overview of Model Testing Scripts for Lab Evaluation
This section documents the scripts used to evaluate different versions of the object detection models in a controlled lab environment. Each script is listed individually, along with a brief explanation of its role in the testing process. The titles provided correspond directly to the filenames of the scripts for easy reference.

## Script for executing the model using the native YOLO library.

``` py linenums="1" title="yolo_object_detection_YOLOLIB.py"
import cv2
from ultralytics import YOLO
import torch
import time

# Set CUDA device
torch.cuda.set_device(0)

# Load the YOLOv8 model
model = YOLO(r"D:\Arybhatta_motors_computer_vision\Yolov8_custom\scripts\models\yolov11l_best.pt")

# Run detection on an image once (optional, just testing)
image_path = r"D:\Arybhatta_motors_computer_vision\Yolov8_custom\scripts\videos_images\istockphoto-174662203-612x612.jpg"
image_results = model(image_path)

# Video path
video_path = r"D:\Arybhatta_motors_computer_vision\Yolov8_custom\scripts\videos_images\WhatsApp Video 2024-07-05 at 01.41.50_72d4a5c5.mp4"
cap = cv2.VideoCapture(video_path)

arr = []  # to store inference times

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success or frame is None:
        break

    # Resize frame if needed
    frame = cv2.resize(frame, (640, 640))

    # Run YOLOv8 detection
    start_time = time.time()
    results = model(frame)
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
```
The script above executes a YOLO model—irrespective of its version—and performs inference on each extracted image frame. Frame extraction, whether from a live camera feed or video stream, is handled using OpenCV. Additionally, the script measures the inference time for every frame; taking the inverse of this value provides the frames per second (FPS).

## Script for executing the YOLO model using OpenCV's DNN module.

``` py linenums="1" title="yolo_object_detection_OPENCV.py"
import cv2
import numpy as np

# Load the frozen graph
net = cv2.dnn.readNet(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\saved_model.pb")

# Specify target device (CPU or GPU)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Initialize video capture object
cap = cv2.VideoCapture(r"E:\Aryabhatta_motors_computer_vision\scripts\videos\WhatsApp Video 2024-07-05 at 01.41.49_9b652ede.mp4")


if not cap.isOpened():
    print("Error: Could not open video.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Prepare input image (resize if necessary)
    frame_resized = cv2.resize(frame, (640, 640))
    blob = cv2.dnn.blobFromImage(frame, 1.0, (640, 640), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Perform inference and get output
    outs = net.forward()

    # Post-process the output (typically, YOLOv3 or similar models)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.99:
                # Calculate bounding box coordinates
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                
                # Draw bounding box
                cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
                
                # Print bounding box coordinates
                print(f"Bounding box coordinates: (left={left}, top={top}, right={left+width}, bottom={top+height})")
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```
This script closely resembles the one mentioned above, but it utilizes a different version of the model. Specifically, the YOLO model is first converted to ONNX format and then to a TensorFlow frozen graph (.pb file). Inference is performed using OpenCV. The purpose of these conversions is to benchmark the model’s performance across various formats to identify the most efficient one.
## Code to run the onnx format of the model

``` py linenums="1" title="yolo_object_detection_onnx.py"
import cv2

from ultralytics import YOLO

import torch

# torch.cuda.set_device(0)

# Load the YOLOv8 model
model = YOLO(r"E:\Aryabhatta_motors_computer_vision\scripts\models\best.onnx")
print("before: ",model.device.type)
results = model(r"E:\Aryabhatta_motors_computer_vision\images_potholes_1\dataset-pothole\yolov8_custom\train\images\01_jpg.rf.3ca97922642224c05e3602b324e899f2.jpg")
#Open the video file
print("after: ",model.device.type)
# The inference on the image is done to utilize the gpu for the inference run on the video, this is a bug in yolo and this method helps us bypass the issue.
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
```
The above code runs inference using the onnx model, for all the reasons mentioned above.

## Script for executing an SSD model in TFLite format using the Mediapipe framework.

Mediapipe is a framework capable of running SSD models. While SSD offers faster performance compared to YOLO, it falls significantly short in terms of accuracy. Use this script if you need to evaluate SSD models. Note that all model files can be found in the /scripts/models directory.
```py title="tflite_mediapipe_detections.py" linenums="1"

import cv2
import numpy as np
import mediapipe as mp
import time

# Load MediaPipe Object Detector
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions

# Path to your TFLite model
model_path = r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\best3.tflite"

# Create Object Detector Options
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    max_results=5,  # Max number of objects per frame
    score_threshold=0.2  # Confidence threshold
)

# Open video file
cap = cv2.VideoCapture(1)

tl = [427, 381]
tr = [752,381]
bl = [161,588]
br = [1029,588]

# Initialize the detector   
with ObjectDetector.create_from_options(options) as detector:
    prev_time = time.time()  # Start time for FPS calculation
    while cap.isOpened():
        ret, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        if not ret:
            break  # End of video
        cv2.circle(frame, tl, 5, (0,0,255,-1))
        cv2.circle(frame, bl, 5, (0,0,255,-1))
        cv2.circle(frame, tr, 5, (0,0,255,-1))
        cv2.circle(frame, br, 5, (0,0,255,-1))
        # cv2.circle(frame, (373, 316), 5, (0,0,255,-1))

        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result_0 = cv2.warpPerspective(frame, matrix, (640,480))
        
        # Convert frame to RGB (MediaPipe format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run object detection
        model_interpreter_start_time = time.time()
        result = detector.detect(mp_image)
        each_interpreter_time = time.time() - model_interpreter_start_time
        
        # Calculate FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time
        
        # Draw results
        if result.detections:
            for detection in result.detections:
                bbox = detection.bounding_box
                x_min, y_min = int(bbox.origin_x), int(bbox.origin_y)
                width, height = int(bbox.width), int(bbox.height)
                x_max, y_max = x_min + width, y_min + height
                print((x_max - x_min)*(y_max - y_min))
                if (x_max - x_min)*(y_max - y_min) < 250000:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    # Draw label & score
                    for category in detection.categories:
                        label = category.category_name
                        score = category.score
                        cv2.putText(result_0, f"{label}: {score:.2f}", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show processed video
        cv2.imshow("Pre_transform", frame)
        cv2.imshow("Object Detection", result_0)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```   