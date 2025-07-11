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

# tl = [503, 243]
# bl = [346, 370]
# tr = [845, 243]  
# br = [1019, 370]

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