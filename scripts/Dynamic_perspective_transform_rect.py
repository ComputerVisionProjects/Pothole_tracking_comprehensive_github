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

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define black color range (adjust as needed)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    # Mask for black regions (detecting tape)
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the tape)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Ensure we have exactly 4 corners
        if len(approx) == 4:
            corners = approx.reshape(4, 2)  # Reshape to (4,2) array
            print("Detected corners:", corners)

            # Draw the corners on the image
            for (x, y) in corners:
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        else:
            print(f"Detected {len(approx)} corners, refining...")

    cv2.imshow("Detected Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()