import numpy as np
import cv2
from Pyfirmata_servo_motor_runner import rotate_servo


K = np.array([[892.86025081, 0, 319.72816385],
                   [0, 894.36852612, 182.8589494 ],
                      [0, 0, 1]])
pixel_point = np.array([373, 316, 1],dtype=np.float32) 



phi = int(input("Input angle: "))
rotate_servo(30+phi)
depth = 94

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


cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        pt_trfm = compute_roi_rotation(phi, pixel_point, depth).astype(int)
        pt_trfm = tuple(map(int, pt_trfm))
        cv2.circle(frame, pt_trfm, 5, (0,0,255,-1))
        

        cv2.imshow('frame', frame) # Initial Capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()    