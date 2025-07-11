
# program to capture single image from webcam in python 
  
# importing OpenCV library 
import cv2
import os
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

import matplotlib
import matplotlib.pyplot as plt
import time

KNOWN_DISTANCE = 50
KNOWN_WIDTH = 14.5
face_width_in_frame = 238

def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image* measured_distance)/ real_width
    return focal_length

def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length)/face_width_in_frame
    return distance

focal_length = FocalLength(KNOWN_DISTANCE, KNOWN_WIDTH, face_width_in_frame)

modelpath = r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\detect1.tflite"
interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()
min_conf=0.5
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


  
# initialize the camera 
# If you have multiple camera connected with  
# current device, assign a value in cam_port  
# variable according to that 
cam_port = 1
cam = cv2.VideoCapture(cam_port) 
  
# reading the input using the camera 
result, image_0 = cam.read() 

if result:
    image = cv2.flip(image_0, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    print(image.shape)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

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
            # if ((ymax-ymin)*(xmax-xmin)) > 50000:
            #     continue
            POTHOLE_WIDTH = xmax - xmin 
            print(POTHOLE_WIDTH)
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
            width_in_image = xmax - xmin
            distance = Distance_finder(focal_length, KNOWN_WIDTH, width_in_image)
            print(distance)

    cv2.imshow("Window", image)
    cv2.imwrite("Depth_sample.jpg",image)
    cv2.waitKey(0)
    cv2.destroyWindow("Window")
