# Script used to develop and evaluate the tracking system and mechanism

This section details the code used for the tracking mechanism in a lab setting. It's important to note that there are two approaches for implementing the tracking system: one involves mounting the camera, while the other involves fixing the camera in place.

## Code used to implement the solution with a mounted camera

The below code is used to implement the mounted camera method, it uses a tensorflow ssd model to run inference.

```py linenums="1" title="tensorflow_ssd_object_detection_OPENCV.py"

# --- Import necessary libraries ---
import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib.pyplot as plt
import time

# --- Import PyFirmata for Arduino control ---
import pyfirmata
from pyfirmata import Arduino, SERVO, util
from time import sleep

# --- Set up Arduino board and define servo pins ---
port = 'COM6'
pin = 10   # Horizontal movement servo
pin1 = 9   # Vertical movement servo
board = pyfirmata.Arduino(port)
board.digital[pin].mode = SERVO
board.digital[pin1].mode = SERVO

# --- Function to rotate a servo motor connected to the given pin ---
def rotate_servo(pin, angle):
    board.digital[pin].write(angle)

# Initialize servo angles
angle = 45 
angle1 = 45
rotate_servo(pin, angle)
rotate_servo(pin1, angle1)

# --- Load TensorFlow Lite model ---
modelpath = r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\detect1.tflite"
interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()
min_conf = 0.55  # Minimum confidence threshold
arr = []  # Store inference times

# --- Get model input/output details ---
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
float_input = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# --- Load label map ---
with open("labelmap.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- Start capturing video ---
video_path = r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\videos\WhatsApp Video 2024-07-05 at 01.41.50_72d4a5c5 (online-video-cutter.com).mp4"
cap = cv2.VideoCapture(1)  # Use webcam (change to 0 or path if needed)

# --- Main loop for processing video frames ---
while cap.isOpened():
    success, frame = cap.read()
    if frame is None:
        break

    if success:
        image = cv2.flip(frame, 1)  # Flip frame horizontally for mirror effect
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        print(image.shape)

        # Resize frame to model's expected input size
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # --- Run inference ---
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        end_time = time.time()

        # Calculate and store inference time
        total_time = end_time - start_time
        arr.append(total_time)
        FPS = str(int(1 / (total_time)))  # Frames per second

        # --- Get model outputs ---
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]    # Bounding boxes
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class indices
        scores = interpreter.get_tensor(output_details[0]['index'])[0]   # Confidence scores

        detections = []

        # --- Loop through detections ---
        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                # Convert bounding box coordinates to image size
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2

                print(x_center, y_center)

                # Draw bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Deadband region is defined
                if (x_center > 300 and x_center < 420) and (y_center > 200 and y_center < 320):
                    continue

                # Servo logic: move left/right based on x position
                if x_center < 320:
                    angle = angle - 1
                    rotate_servo(pin, angle)
                if x_center > 320:
                    angle = angle + 1
                    rotate_servo(pin, angle)

                # Servo logic: move up/down based on y position
                if y_center < 240:
                    angle1 = angle1 - 1
                    rotate_servo(pin1, angle1)
                if y_center > 240:
                    angle1 = angle1 + 1
                    rotate_servo(pin1, angle1)

                # Prepare and draw label
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Append detection details
                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

        # Display FPS on frame
        cv2.putText(image, f"FPS = {FPS}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (178, 255, 102), 2)
        
        # Show the annotated frame
        cv2.imshow('annotated frame', image)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
```
The code above adjusts the position of the mechanism by comparing the center of the bounding box of the pothole with the center of the camera. It corrects the position by calculating the error in each iteration and gradually reducing it. However, the issue with this approach is that the algorithm may cause the mechanism to overshoot its target.

## Code used to implement the stationary camera solution

```py linenums="1" title="yolo_final_implementation_optimized.py"

import cv2
import numpy as np
from ultralytics import YOLO
from Equation_solver_closed_form import tangent_point_finder
import math
import time
from esp32_wifi import send_servo_angles
import torch
from threading import Thread

def send_angles_async(a1, a2):
    Thread(target=send_servo_angles, kwargs={'s1': a1, 's2': a2}, daemon=True).start()

KNOWN_DISTANCE = 50
KNOWN_WIDTH = 14.5
face_width_in_frame = 238

# Perspective transform corners (only set once)
tl = [336, 434]
tr = [862, 427]
bl = [124, 578]
br = [1143, 559]
pts1 = np.float32([tl, bl, tr, br])
pts2 = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_transformed.avi', fourcc, 20.0, (1280, 720))

# Constants
H = 90
scale_factor_x = 280 / 1280
scale_factor_y = 240 / 720
hlf_width_pprgn = 140
vertical_width_pprgn = 240

torch.cuda.set_device(0)
model = YOLO("D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\models\yolov11n_best_25k.pt")

cam = cv2.VideoCapture(2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_frame_time = 0


def send_angles_async(a1, a2):
    Thread(target=send_servo_angles, kwargs={'s1': a1, 's2': a2}, daemon=True).start()

if not cam.isOpened():
    print("Error: Could not load camera")
    exit()

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 0)
    for pt in [tl, bl, tr, br]:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    
    frame = cv2.flip(frame, 1)
    image_resized = frame

    # Perspective transform
    warped = cv2.warpPerspective(image_resized, matrix, (1280, 720))
    

    # Inference
    start_infer = time.time()
    results = list(model.predict(warped, conf=0.2, device="cuda", stream=True, verbose=False))
    print("Inference time:", time.time() - start_infer)

    if not results:
        cv2.imshow("YOLOv8 Stream", warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    result = results[0]
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        y_centers, rect_sizes, coords = [], [], []

        for box in boxes:
            coords_np = box.xyxy[0].detach().cpu().numpy()
            if coords_np.shape[0] != 4:
                continue
            x1, y1, x2, y2 = coords_np
            x_center = float((x1 + x2) / 2)
            y_center = float((y1 + y2) / 2)
            area = (x2 - x1) * (y2 - y1)

            y_centers.append(y_center)
            rect_sizes.append(area)
            coords.append([x1, y1, x2, y2])

        if coords:
            coords = np.array(coords)
            y_arr = np.array(y_centers)
            size_arr = np.array(rect_sizes)
            weights = (y_arr / 480) * 0.2 + (size_arr / (640 * 480)) * 0.8
            idx = np.argmax(weights)

            x1, y1, x2, y2 = coords[idx]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            warped = cv2.rectangle(warped, (int(x1), int(y1)), (int(x2), int(y2)), (255, 248, 150), 4)

            # FPS calc
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time + 1e-8)
            prev_frame_time = new_frame_time
            cv2.putText(warped, f"FPS: {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Angle calculations
            x_coord = scale_factor_x * x_center
            y_coord = scale_factor_y * y_center
            x_trans = x_coord - hlf_width_pprgn
            y_trans = 250 + (vertical_width_pprgn - y_coord)
            ox = x_trans + 6.5
            oy = y_trans + 1

            ang1 = math.degrees(math.atan(H / math.sqrt(ox ** 2 + oy ** 2)))
            ang2 = math.degrees(math.atan(ox / oy))

            motor1 = 90 - ang1
            motor2 = 90 - ang2 + 3

            try:
                send_angles_async(motor2, motor1)
                print(f"Motor1: {round(motor1)}, Motor2: {round(motor2)}")
            except Exception as e:
                print("Servo communication failed:", e)

        cv2.imshow("YOLOv8 Stream", warped)
        cv2.imshow("original", image_resized)
    out.write(warped)
    cv2.imshow("YOLOv8 Stream", warped)
    cv2.imshow("original", image_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
cam.release()
cv2.destroyAllWindows()

```
This script implements a real-time pothole detection and servo control system using computer vision and embedded communication. The key components of the pipeline are as follows:

Video Acquisition & Perspective Correction:
A live video stream is captured from a camera and passed through a homography-based perspective transform to simulate a top-down orthographic view. This improves spatial accuracy for downstream calculations.

Object Detection Using YOLO:
A YOLOv8 model, pre-trained and optimized for pothole detection, performs inference on each frame using GPU acceleration. Detection confidence is thresholded at 0.2 to filter out low-confidence predictions.

Target Prioritization:
Among multiple detected bounding boxes, a weighted scoring strategy is applied—favoring larger and lower-positioned boxes (indicative of proximity)—to select the most relevant target for actuation.

Kinematic Angle Computation:
The pixel coordinates of the selected target are mapped to a scaled coordinate space. From these, two servo angles are computed using basic trigonometric relationships to determine the orientation required to align with the detected object.

Asynchronous Servo Communication via ESP32:
The computed angles are sent to an ESP32 microcontroller over WiFi using a non-blocking thread. This ensures real-time responsiveness without interrupting the main inference and visualization loop.

Visualization & Logging:
The processed video frames—annotated with bounding boxes, FPS, and servo data—are displayed in real time and optionally saved to disk for post-analysis.

This architecture enables responsive and spatially accurate pothole tracking suitable for autonomous or semi-autonomous robotic systems.

Keep in mind I observed even when the camera/apparatus was tilted the error introduced was not significant enough to offset the light path from the pothole, I would suggest anyone working on this project to work on the fixed camera solution only.

## PID control system for the mounted camera setup

PID is a control method that allows us to manage various aspects of motor motion. Below is the code for implementing the PID control system.

```py title="PID_camera_tracking.py" linenums="1"

# Import required libraries
import cv2
import numpy as np
import time
import serial
from simple_pid import PID
from ultralytics import YOLO  # YOLO from Ultralytics
import torch

# Select the CUDA device (GPU)
torch.cuda.set_device(0)

# Load YOLOv11n model - replace with your trained model path
model = YOLO(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\best-yolov11n.pt")
print("before: ", model.device.type)

# Test run on a sample image to ensure the model loads correctly
results = model(r"E:\Aryabhatta_motors_computer_vision\images_potholes\78778.png")
print("after: ", model.device.type)

# Setup serial communication with Arduino (adjust COM port accordingly)
arduino = serial.Serial(port='COM7', baudrate=9600, timeout=1)
time.sleep(2)  # Give some time to establish the serial connection

# Initialize PID controllers for X and Y servos (horizontal and vertical)
pid_x = PID(0.02, 0, 0, setpoint=0)
pid_y = PID(0.02, 0, 0, setpoint=0)

# Set servo angle limits
pid_x.output_limits = (-45, 45)
pid_y.output_limits = (-45, 45)

# Deadband (tolerance) in pixels - no correction if within this range
deadband_x = 10
deadband_y = 15

# Start video capture (adjust the index based on camera setup)
cap = cv2.VideoCapture(1)

# Initial angles for the servo motors
servo_angle_x = 90  # Mid position for horizontal
servo_angle_y = 40  # Some default vertical position

def send_servo_command(servo, angle):
    """ Send servo angle command to Arduino over serial """
    command = f"{servo}:{int(angle)}\n"
    arduino.write(command.encode())
    time.sleep(0.02)  # Small delay for serial communication stability

# Send the initial servo position
send_servo_command("X", servo_angle_x)
send_servo_command("Y", servo_angle_y)

try:
    while True:
        # Read a frame from the camera and horizontally flip it
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        # Run tracking with confidence threshold
        results = model.track(frame, persist=True, conf=0.5)

        # Set box color for annotation
        color = (255, 248, 150)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                object_center_x = (x1 + x2) / 2
                object_center_y = (y1 + y2) / 2

                # Calculate center of the frame
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2

                # Compute X and Y errors from center
                error_x = object_center_x - frame_center_x
                error_y = object_center_y - frame_center_y

                # If horizontal error exceeds tolerance, correct with PID
                if abs(error_x) > deadband_x:
                    error_x = error_x.cpu().numpy()
                    correction_x = pid_x(error_x)
                    servo_angle_x = np.clip(servo_angle_x - correction_x, 0, 180)
                    print(servo_angle_x)
                    send_servo_command("X", servo_angle_x)

                # If vertical error exceeds tolerance, correct with PID
                if abs(error_y) > deadband_y:
                    error_y = error_y.cpu().numpy()
                    correction_y = pid_y(error_y)
                    servo_angle_y = np.clip(servo_angle_y - correction_y, 0, 60)
                    print(servo_angle_y)
                    send_servo_command("Y", servo_angle_y)

        # Display the annotated output
        annotated_frame = results[0].plot()
        cv2.imshow("Tracking", annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# On exit, release camera and serial resources
finally:
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
```

The deadband refers to the range where the mechanism won't move further if it's pointing within it. PID is a system aimed at minimizing error and determining the best way to reduce it, considering factors like speed, smoothness, and stability. The P (proportional) component controls how quickly the error is reduced, the I (integral) component addresses accumulated error over time, enhancing pothole detection accuracy, and the D (derivative) component stabilizes the motion, preventing overshooting. In our system, a high P value is necessary to match the speed at which the bike moves.

