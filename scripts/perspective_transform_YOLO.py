import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\best1.pt")

print("before: ",yolo_model.device.type)

results = yolo_model(r"E:\Aryabhatta_motors_computer_vision\images_potholes\78778.png")

print("after: ",yolo_model.device.type)
# Define points for perspective transform
tl = [550, 338]
bl = [190, 487]
tr = [830, 338]  
br = [1164, 487]

cap = cv2.VideoCapture(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\videos\bumpy_bike_video.mp4")

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel position: ({x}, {y})")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    cv2.circle(frame, tuple(tl), 5, (0, 0, 255), -1)
    cv2.circle(frame, tuple(bl), 5, (0, 0, 255), -1)
    cv2.circle(frame, tuple(tr), 5, (0, 0, 255), -1)
    cv2.circle(frame, tuple(br), 5, (0, 0, 255), -1)
    cv2.circle(frame, (373, 316), 5, (0, 0, 255), -1)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result1 = cv2.warpPerspective(frame, matrix, (640, 480))
    
    # Perform YOLO inference on the perspective-transformed image
    results = yolo_model(result1, conf = 0.1)
    
    # Draw YOLO detections on transformed frame
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(result1, (x1, y1), (x2, y2), (0, 255, 0), 2)
     
    # Display frames
    cv2.imshow('frame', frame)  # Initial Capture
    cv2.imshow('frame1', result1)  # Transformed Capture with YOLO Inference
    cv2.setMouseCallback("frame", click_event)
 
    if cv2.waitKey(24) == 27:
        break
 
cap.release()
cv2.destroyAllWindows()