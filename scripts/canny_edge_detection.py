import cv2
import numpy as np

# Load the image
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (or use Canny edge detection)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("contour_boundary", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()