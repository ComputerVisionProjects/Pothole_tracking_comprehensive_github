import numpy as np
import cv2
from Pyfirmata_servo_motor_runner import rotate_servo
import time
# Camera intrinsic matrix (K) [Replace with your actual values]
K = np.array([[892.86025081, 0, 319.72816385],
                   [0, 894.36852612, 182.8589494 ],
                      [0, 0, 1]])

# Original 4 pixel coordinates (top-left, bottom-left, top-right, bottom-right, )
pixel_coords = np.array([[370, 265, 1], [485, 216, 1], [399, 368, 1], [540, 307, 1]], dtype=np.float32) 
     # Shape (3,4)

rotate_servo(30)
time.sleep(3)
cap = cv2.VideoCapture(1)
Flag, image = cap.read()
image = cv2.flip(image, 1)
cv2.circle(image, [370, 265], 5, (0,0,255,-1))
cv2.circle(image, [485, 216], 5, (0,0,255,-1))
cv2.circle(image, [399, 368], 5, (0,0,255,-1))
cv2.circle(image, [540, 307], 5, (0,0,255,-1))
cv2.imwrite("before_ppdynm.png", image)
i = 0
phi = int(input("Input angle: "))
rotate_servo(30+phi)
depth = [116, 91, 116, 91]


def compute_roi_rotation(phi, pixel_point, z):
    K_inv = np.linalg.inv(K)  # Inverse of the intrinsic matrix
    world_point = K_inv @ (pixel_point * z)  # Scale by depth
    world_point = np.append(world_point, 1)  # Convert to homogeneous (x, y, z, 1)

    # Define rotation matrix for camera tilt along X-axis (e.g., 10 degrees)
    theta = np.deg2rad(phi)  # Convert degrees to radians
    R_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]], dtype=np.float32)  # Homogeneous 4x4 matrix

    # Step 2: Apply rotation
    world_point_tilted = R_x @ world_point

    # Step 3: Convert back to pixel coordinates
    pixel_coords_tilted = K @ world_point_tilted[:3]  # Ignore homogeneous 1
    pixel_coords_tilted /= pixel_coords_tilted[2]  # Normalize

    # Output new pixel position
    print(pixel_coords_tilted[:2])
    return pixel_coords_tilted[:2]

while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        pts = []
        for pt in pixel_coords:
            print(pt)
            pt_trfm = compute_roi_rotation(phi, pt, depth[i]).astype(int)
            pts.append(pt_trfm)
            i += 1
        i = 0
        tl = pts[0]
        bl = pts[1]
        tr = pts[2]
        br = pts[3]

        cv2.circle(frame, tl, 5, (0,0,255,-1))
        cv2.circle(frame, bl, 5, (0,0,255,-1))
        cv2.circle(frame, tr, 5, (0,0,255,-1))
        cv2.circle(frame, br, 5, (0,0,255,-1))

        pts1 = np.float32(pts)
        pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, (640,480))
        
        # Wrap the transformed image
        cv2.imshow('frame', frame) # Initial Capture
        cv2.imshow('frame1', result) # Transformed Capture
        # cv2.setMouseCallback("frame", click_event)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("tranformed.png", frame)
            break
    
cap.release()
cv2.destroyAllWindows()

