import cv2
import numpy as np


tl = [148, 275]       
tr = [408, 274]  
bl = [28, 359]
br = [549, 333]     
   
output_width = 640
output_height = 480 
# Initialize webcam

cap = cv2.VideoCapture(2)  # Use 0 or the index of your webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Define codec and output file
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_transformed.avi', fourcc, 20.0, (1280, 720))

# Define perspective transform source and destination points
src_pts = np.float32([tl, tr, bl, br])
dst_pts = np.float32([
        [0, 0],                     # Top-left
        [output_width, 0],          # Top-right
        [0, output_height],        # Bottom-left
        [output_width, output_height] # Bottom-right
    ])
    

# Compute the perspective transform matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame,0)
    if not ret:
        break

    # Resize to 1280x720 if not already
    frame_resized = cv2.resize(frame, (1280, 720))

    # Apply perspective transform
    warped = cv2.warpPerspective(frame_resized, M, (640, 480))
   
    # Write the frame
    out.write(warped)

    # Display the result (optional)
    cv2.imshow('Transformed Video', frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
