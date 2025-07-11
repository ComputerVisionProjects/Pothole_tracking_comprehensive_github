import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO(r"D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\models\yolov11n_best_25k.pt")

# Define points (order: top-left, top-right, bottom-left, bottom-right)
tl = [336, 434]       
tr = [862, 427]  
bl = [124, 578]
br = [1143, 559]    

cap = cv2.VideoCapture("D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\Videos\output_transformed(1) (online-video-cutter.com) (1).mp4")  # Or your URL
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel position: ({x}, {y})")

# Define output size (width, height)
output_width = 640
output_height = 480 # Or adjust to match your desired aspect ratio

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # frame = cv2.flip(frame,0)
    frame_resized = cv2.resize(frame, (1280, 720))
    for pt in [tl, bl, tr, br]:
        cv2.circle(frame_resized, pt, 5, (0, 0, 255), -1)

    # Create source and destination points
    pts1 = np.float32([tl, tr, bl, br])
    pts2 = np.float32([
        [0, 0],                     # Top-left
        [output_width, 0],          # Top-right
        [0, output_height],        # Bottom-left
        [output_width, output_height] # Bottom-right
    ])
    
    # Get perspective transform and warp
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(frame_resized, matrix, (output_width, output_height))
    warped1 = cv2.resize(warped,(640,360))
    # Run YOLO detection (uncomment when ready)
    results = model(warped1, conf=0.3)
    annotated = results[0].plot() 
    cv2.imshow("Original", frame_resized)
    cv2.imshow("Top-Down View",annotated)
    cv2.setMouseCallback("Original", click_event)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()