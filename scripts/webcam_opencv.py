import cv2

cap = cv2.VideoCapture(2)  # try 0, 1, 2 if needed

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,0)
    if not ret:
        break
    cv2.imshow('Phone Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
