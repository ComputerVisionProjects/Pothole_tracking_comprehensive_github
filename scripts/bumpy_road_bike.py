import cv2


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel position: ({x}, {y})")
 

frame = cv2.imread(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\captured_image.jpg")




while True:
    cv2.imshow("Camera Feed", frame)
    cv2.setMouseCallback("Camera Feed", click_event)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
