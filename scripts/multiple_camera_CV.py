import cv2

# Initialize two camera sources (change index 0 and 1 as per your system)
cap1 = cv2.VideoCapture(1)  
cap2 = cv2.VideoCapture(3)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

while True:
    # Capture frame-by-frame
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Failed to capture image")
        break

    # Display the resulting frames
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture objects
cap1.release()
cap2.release()
cv2.destroyAllWindows()
