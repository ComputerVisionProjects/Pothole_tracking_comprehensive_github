import cv2
import numpy as np

# Load the ONNX model
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