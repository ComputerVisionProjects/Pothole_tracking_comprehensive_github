# Code used to implement and evaluate the perspective transformation

In this section, I will explain the code used to detect and test the perspective transformation.

## Code used to determine the coordinates of the corners of the flat surface.

The following code is used to identify the corner coordinates of a flat rectangular surface. This step is crucial for accurately applying the perspective transform to the surface when calculating the pothole coordinates. The code also includes sections for applying the perspective transform, but these can be disregarded here, as the code can also be used to verify the perspective transform.

```py title="Perspective_transform_coord_finder.py" linenums="1"
import cv2 
import numpy as np 
 
# === Turn on Laptop's webcam ===
# Index '3' selects the specific webcam device; change if needed (0 is usually default).
cap = cv2.VideoCapture(3)

# === Define four corner points in the original camera frame for perspective transformation ===
# These are manually chosen and represent a quadrilateral area that will be transformed to a rectangle.
tl = [224, 68]     # Top-left corner
bl = [5, 223]      # Bottom-left corner
tr = [638, 149]    # Top-right corner
br = [565, 420]    # Bottom-right corner

# === Callback function to capture mouse clicks ===
# Prints the pixel coordinates where the user clicks on the image window.
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel position: ({x}, {y})")

# === Main loop to continuously capture frames from the webcam ===
while True:
    ret, frame = cap.read()  # Capture a single frame

    # Display the raw frame before any processing
    cv2.imshow('frame', frame)

    # === Draw red circles on the perspective transformation corners for visual reference ===
    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    # Draw another reference point (optional, might be a target or calibration marker)
    cv2.circle(frame, (373, 316), 5, (0, 0, 255), -1)

    # === Prepare points for perspective transformation ===
    pts1 = np.float32([tl, bl, tr, br])  # Source points (distorted quadrilateral)
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])  # Destination points (rectangle)

    # === Compute the perspective transformation matrix ===
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # === Apply the perspective warp using the matrix ===
    result = cv2.warpPerspective(frame, matrix, (640, 480))

    # === Show the frame after transformation ===
    # `frame1` is still showing the original frame here due to the line below,
    # likely a typo — should be `cv2.imshow('frame1', result)` if intent is to show transformed image.
    cv2.imshow('frame1', frame) 

    # Enable mouse click tracking on the original frame window
    cv2.setMouseCallback("frame", click_event)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# === Release camera and close all OpenCV windows ===
cap.release()
cv2.destroyAllWindows()

```
The code above captures mouse clicks and displays the pixel coordinates of the clicked locations. This allows us to easily identify the pixel coordinates of the corners by simply clicking on them in the camera image.

## Code used to apply a dynamic perspective transformation

This code establishes a system that identifies the four corners of a rectangular area and applies a perspective transformation to it.

```py title="Dynamic_perspective_transform_rect.py" linenums="1"

import cv2
import numpy as np

# Initialize the camera stream
cap = cv2.VideoCapture(0)

# Edge case handling
if not cap.isOpened():
    print("Error: Could not open video.")

# Loop to read the camera stream
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
```

The code above is used to identify a rectangular area marked by black tapes. The region is then detected using contour detection, which was done for specific technical reasons.
