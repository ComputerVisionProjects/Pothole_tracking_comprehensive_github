import cv2
import numpy as np

# Load your camera matrix and distortion coefficients
# You need to calibrate your camera beforehand to get these values
mtx = np.array([[892.86025081, 0, 319.72816385],
                   [0, 894.36852612, 182.8589494 ],
                      [0, 0, 1]])
dist = np.array([-2.17954003e-02, 3.78637782e-01, 4.81147467e-04, 1.20099914e-03,
  -8.01539551e-01])  # Replace with actual values

# Open webcam feed
cap = cv2.VideoCapture(1)

# Get frame dimensions
ret, frame = cap.read()
h, w = frame.shape[:2]

# Compute the optimal new camera matrix and undistort maps
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, (w, h), 5)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Undistort the frame
    undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    
    # Display the result
    cv2.imshow("distorted_frame", frame)
    cv2.imshow("Undistorted Feed", undistorted_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
