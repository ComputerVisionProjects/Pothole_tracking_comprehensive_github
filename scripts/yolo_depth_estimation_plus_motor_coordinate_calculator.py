import cv2
import numpy as np 
from ultralytics import YOLO
from Equation_solver_closed_form import tangent_point_finder
import math
import time
import requests
from esp32_wifi import send_servo_angles
import torch
from threading import Thread
# from Pyfirmata_servo_motor_runner import rotate_servo

def send_angles_async(a1, a2):
    Thread(target=send_servo_angles, kwargs={'s1': a1, 's2': a2}, daemon=True).start()

KNOWN_DISTANCE = 50
KNOWN_WIDTH = 14.5
face_width_in_frame = 238

tl = [336, 434]       
tr = [862, 427]  
bl = [124, 578]
br = [1143, 559]      

count = 0
prev_frame_time = 0
new_frame_time = 0

x_center1 = 0
y_center1 = 0
box_coord = []    
y_center_arr = []
rect_size_arr = []
distance = 200
# Radius = 3.319
H = 90
scale_factor_x = 280/1280
scale_factor_y = 240/720
hlf_width_pprgn = 140
vertical_width_pprgn = 240
angle = 90
angle1 = 90

torch.cuda.set_device(0)

# Load the YOLOv8 model
model = YOLO(r"D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\models\yolov11n_best_25k.pt")
print(f"before: {model.device}")
# Run detection on an image once (optional, just testing)
image_path = r"D:\Arybhatta_motors_computer_vision\Yolov8_custom\scripts\videos_images\istockphoto-174662203-612x612.jpg"

image_results = model(image_path) 
print(f"After: {model.device}")

#capture image
cam_port = 1
cam = cv2.VideoCapture(2)   
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# reading the input using the camera
if not cam.isOpened():
    print("Error could not load camera") 

while cam.isOpened():
    result, image = cam.read()   
    frame_resized = cv2.resize(image, (1280, 720))
    if result:
        image = cv2.flip(image, 0)
        imH, imW, _ = image.shape
        print(image.shape)
        image_resized = image
        cv2.circle(image_resized, tl, 5, (0,0,255,-1))
        cv2.circle(image_resized, bl, 5, (0,0,255,-1))
        cv2.circle(image_resized, tr, 5, (0,0,255,-1))
        cv2.circle(image_resized, br, 5, (0,0,255,-1))

        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0,0], [0,720], [1280,0], [1280,720]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result_0 = cv2.warpPerspective(image_resized, matrix, (640,480))
        result_0 = cv2.flip(result_0, 1)                 
        # Run inference on an image
        confidence = 0.2
        result = model.predict(image_resized, conf = confidence)
        boxes = result[0].boxes
        color = (255, 248, 150) 
        box_coord = []
        rect_size_arr = []
        y_center_arr = []
        if boxes:
            for box in boxes:
                print(box.xyxy[0])
                x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates in format [x1, y1, x2, y
                print("Box coordinates:", x1, y1, x2, y2)
                x_center = float((x1+x2)/2)
                y_center = float((y1+y2)/2)    
                y_center_arr.append(y_center)
                rect_size = (x2 - x1)*(y2 - y1)
                rect_size_arr.append(rect_size)
                box_coord.append([x1, y1, x2, y2])
            combined_arr = list(zip(y_center_arr, rect_size_arr))
            weight_arr = []
            for x in range(len(combined_arr)):
                weight = ((combined_arr[x][0])/480)*0.2 + (combined_arr[x][1]/(640*480))*0.8
                weight_arr.append(weight)
            pothole_weight = max(weight_arr)
            index = weight_arr.index(pothole_weight)
            x1, y1, x2, y2 = box_coord[index]
            x_center = (x1 + x2)/2
            y_center = (y1 + y2)/2
            motor_angles = []
            weight_arr = []
            box_coord = []
            y_center_arr = []
            rect_size_arr = []
            image_resized = cv2.rectangle(image_resized, (int(x1),int(y1)), (int(x2),int(y2)), color , thickness = 4)
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time + 1e-8)  # Add epsilon to avoid division by zero
            prev_frame_time = new_frame_time

            # Overlay FPS on result_0 image
            fps_text = f"FPS: {int(fps)}"   
            cv2.putText(result_0, fps_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(x_center,y_center)
            x_coordinate =scale_factor_x*x_center             
            y_coordinate = scale_factor_y*y_center 
            x_coordinate_trfm = x_coordinate - hlf_width_pprgn  # add correction to get corodinates from the camera origin
            y_coordinate_trfm = distance + (vertical_width_pprgn - y_coordinate)
            print(f"x cordinate in cm is {x_coordinate_trfm}, y coordinate in cm is {y_coordinate_trfm}") 
            origin_shift_x = x_coordinate_trfm + 6.5
            origin_shift_y = y_coordinate_trfm + 1 
            servo_motor1_angle_radians = math.atan(H/math.sqrt(origin_shift_x**2 + origin_shift_y**2))
            servo_motor2_angle_radians = math.atan(origin_shift_x/origin_shift_y)
            servo_motor1_angle_degrees = math.degrees(servo_motor1_angle_radians)
            servo_motor2_angle_degrees = math.degrees(servo_motor2_angle_radians)
            print(f"servo_motor1_angle_degrees={servo_motor1_angle_degrees},servo_motor2_angle_degrees={servo_motor2_angle_degrees}")
            motor2_angle = 90 - servo_motor2_angle_degrees + 3
            motor1_angle = 90 - servo_motor1_angle_degrees + 7
            try:
                send_angles_async(motor2_angle, motor1_angle)
                print(f"Motor1: {round(motor1)}, Motor2: {round(motor2)}")
            except Exception as e:
                print("Servo communication failed:", e)     
            print(f"motor 1 angle is {round(motor1_angle)},motor 2 angle is {round(motor2_angle)}")
            # with open("log.txt", "a") as file:
            #     file.write(f"Pothole no: {count}, Pothole coordinates: {x_center,y_center}, confidence={confidence}\n")
            # count += 1    
        cv2.imshow("yolov8_testing_stationarycam" , image_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
        
cam.release()
cv2.destroyAllWindows()        
