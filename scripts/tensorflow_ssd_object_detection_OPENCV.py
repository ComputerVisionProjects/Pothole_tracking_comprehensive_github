import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

import matplotlib
import matplotlib.pyplot as plt
import time

import pyfirmata
from pyfirmata import Arduino, SERVO, util
from time import sleep
port = 'COM6'
pin = 10 
pin1 = 9
board = pyfirmata.Arduino(port)
board.digital[pin].mode = SERVO
board.digital[pin1].mode = SERVO
def rotate_servo(pin,angle):
    board.digital[pin].write(angle)

angle = 45 
angle1 = 45

rotate_servo(pin,angle)
rotate_servo(pin1, angle)
# Load TFLite model and allocate tensors
modelpath = r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\detect1.tflite"
interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()
min_conf=0.55
arr = []

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
# Load label map
float_input = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

with open("labelmap.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

video_path = r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\videos\WhatsApp Video 2024-07-05 at 01.41.50_72d4a5c5 (online-video-cutter.com).mp4"
cap = cv2.VideoCapture(1)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if frame is None:
        break
    if success:    
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        print(image.shape)
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        end_time = time.time()
        total_time = end_time - start_time
        arr.append(total_time)
        FPS = str(int(1/(total_time)))
        # Retrieve detection results

        boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
    
        detections = []

        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                x_center = (xmin+xmax)/2
                y_center = (ymin+ymax)/2
                print(x_center, y_center)
                # if ((ymax-ymin)*(xmax-xmin)) > 50000:
                #     continue 
                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                if (x_center > 300 and x_center < 420) and (y_center > 200 and y_center < 320):
                    continue
                if x_center < 320:
                    angle = angle - 1
                    rotate_servo(pin, angle)
                if x_center > 320:
                    angle = angle + 1
                    rotate_servo(pin, angle)
                if y_center < 240:
                    angle1 = angle1 - 1
                    rotate_servo(pin1, angle1)  
                if y_center > 240:
                    angle1 = angle1 + 1
                    rotate_servo(pin1, angle1)             
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
        cv2.putText(image ,f"FPS = {FPS}",(50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(178,255,102), 2)
        cv2.imshow('annotated frame', image) 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()       
cv2.destroyAllWindows()                 
