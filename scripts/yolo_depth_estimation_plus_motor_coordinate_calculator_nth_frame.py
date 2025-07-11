import cv2
import numpy as np 
from ultralytics import YOLO
from Equation_solver_closed_form import tangent_point_finder
import math
import time
import requests
from esp32_wifi import send_servo_angles
import torch

# Constants
KNOWN_DISTANCE = 50
KNOWN_WIDTH = 14.5
face_width_in_frame = 238
tl = [451, 420]       
tr = [1164, 440]  
bl = [195, 552]
br = [1413, 601]    
prev_frame_time = 0
new_frame_time = 0
x_center1 = 0
y_center1 = 0
box_coord = []    
y_center_arr = []
rect_size_arr = []
distance = 350
H = 90
scale_factor_x = 280/1280
scale_factor_y = 175/720
hlf_width_pprgn = 140
vertical_width_pprgn = 87.5
angle = 90
angle1 = 90
url = 'http://192.0.0.4:8080/video'

torch.cuda.set_device(0)

# Load YOLO model
model = YOLO(r"D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\models\garage_pothole_1.pt")
print(f"Model loaded on: {model.device}")

# Initialize cameras using DirectShow backend to avoid MSMF issues
cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cam1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not (cam.isOpened() and cam1.isOpened()):
    print("Error: Could not open camera(s)")
    exit()

SKIP_FRAMES = 5
frame_count = 0

while cam.isOpened() and cam1.isOpened():
    # Skip SKIP_FRAMES - 1 frames
    for _ in range(SKIP_FRAMES - 1):
        cam.read()
        cam1.read()

    result, image = cam.read()
    result1, image_1 = cam1.read()

    if not result or not result1:
        print("Warning: Failed to grab frame.")
        continue

    image_1 = cv2.flip(image_1, 1)
    imH, imW, _ = image.shape
    image_resized = image
    image_resized1 = cv2.resize(image_1, (320, 320))

    # Draw corner points
    for pt in [tl, bl, tr, br]:
        cv2.circle(image_resized, pt, 5, (0, 0, 255), -1)

    # Perspective transform
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result_0 = cv2.warpPerspective(image_resized, matrix, (1280, 720))

    # YOLO detection on every 5th frame
    result = model.predict(result_0, conf=0.2, device="cuda")
    boxes = result[0].boxes
    color = (255, 248, 150) 

    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x_center = float((x1 + x2) / 2)
            y_center = float((y1 + y2) / 2)
            y_center_arr.append(y_center)
            rect_size = (x2 - x1) * (y2 - y1)
            rect_size_arr.append(rect_size)
            box_coord.append([x1, y1, x2, y2])

        # Weighting
        weight_arr = []
        for i in range(len(y_center_arr)):
            weight = (y_center_arr[i] / 480 * 0.2) + (rect_size_arr[i] / (640 * 480) * 0.8)
            weight_arr.append(weight)

        index = weight_arr.index(max(weight_arr))
        x1, y1, x2, y2 = box_coord[index]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        result_0 = cv2.rectangle(result_0, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)

        # FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time + 1e-8)
        prev_frame_time = new_frame_time
        cv2.putText(result_0, f"FPS: {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Coordinate conversion
        x_coordinate = scale_factor_x * x_center             
        y_coordinate = scale_factor_y * y_center 
        x_coordinate_trfm = x_coordinate - hlf_width_pprgn
        y_coordinate_trfm = distance + (vertical_width_pprgn - y_coordinate)
        print(f"x cm: {x_coordinate_trfm}, y cm: {y_coordinate_trfm}")

        # Servo angle computation
        origin_shift_x = x_coordinate_trfm + 6.5
        origin_shift_y = y_coordinate_trfm + 1
        servo_motor1_angle = math.degrees(math.atan(H / math.sqrt(origin_shift_x**2 + origin_shift_y**2)))
        servo_motor2_angle = math.degrees(math.atan(origin_shift_x / origin_shift_y))
        motor2_angle = 90 - servo_motor2_angle - 3
        motor1_angle = 90 - servo_motor1_angle + 5

        print(f"Motor 1 angle: {round(motor1_angle)}, Motor 2 angle: {round(motor2_angle)}")
        # send_servo_angles(s1=motor2_angle, s2=motor1_angle)

        # Reset arrays
        y_center_arr.clear()
        rect_size_arr.clear()
        box_coord.clear()
        weight_arr.clear()

    # Show outputs
    cv2.imshow("Stationary Cam", result_0)
    cv2.imshow("Moving Cam", image_resized1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cam1.release()
cv2.destroyAllWindows()
