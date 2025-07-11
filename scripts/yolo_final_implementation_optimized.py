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
