import cv2

# Replace with your Android IP and port
url = 'http://192.168.107.36:8080/video'

# Open the video stream
cap = cv2.VideoCapture(2)


if not cap.isOpened():
    print("Failed to connect to the IP webcam.")
    exit()

while True:
    ret, frame = cap.read()
    print(frame.shape)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if not ret:
        print("Failed to get frame.")
        break

    cv2.imshow('Android IP Webcam', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
