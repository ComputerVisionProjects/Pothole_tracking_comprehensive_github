import cv2

# Initialize the camera
cap = cv2.VideoCapture(r"d:\Aryabhatta_computer_vision\Yolov8_custom\scripts\videos\bumpy_bike_video.mp4")

# Global variable to store the latest frame
latest_frame = None

def capture_image(event, x, y, flags, param):
    global latest_frame
    if event == cv2.EVENT_LBUTTONDOWN and latest_frame is not None:
        cv2.imwrite("captured_image.jpg", latest_frame)
        print("Image captured and saved as captured_image.jpg")

cv2.namedWindow("Camera Feed")
cv2.setMouseCallback("Camera Feed", capture_image)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    latest_frame = frame.copy()  # Update the latest frame

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()