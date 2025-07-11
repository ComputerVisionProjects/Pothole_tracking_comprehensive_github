import serial
import time
import math
import numpy as np
import cv2

# Serial setup (Change "COM7" to your port, or "/dev/serial0" for Raspberry Pi)
ser = serial.Serial("COM7", 115200, timeout=1)
time.sleep(2)

pt = np.array([300, 450, 1], dtype=np.float32)
K = np.array([[892.86025081, 0, 319.72816385],
              [0, 894.36852612, 182.8589494],
              [0, 0, 1]])

cap = cv2.VideoCapture(1)  # Use first available camera 
time.sleep(2)  # Allow camera to warm up

ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from camera")
    cap.release()
    exit()

image_height, image_width = frame.shape[:2]
x0, y0 = image_width // 2, image_height // 2  # Start at center

def compute_roi_rotation(phi, pixel_point, z):
    K_inv = np.linalg.inv(K)  # Inverse of intrinsic matrix
    world_point = K_inv @ (pixel_point * z)  # Scale by depth
    world_point = np.append(world_point, 1)  # Convert to homogeneous (x, y, z, 1)

    # Define rotation matrix for roll along Z-axis
    theta = np.deg2rad(phi)
    R_z = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                    [np.sin(theta), np.cos(theta), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.float32)

    world_point_tilted = R_z @ world_point
    pixel_coords_tilted = K @ world_point_tilted[:3]
    pixel_coords_tilted /= pixel_coords_tilted[2]  # Normalize

    return pixel_coords_tilted[:2]

while True:
    frame = cv2.flip(frame, 1)
    try:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            if data:
                yaw, roll, pitch, _, _, _ = map(float, data.split(","))
                
                roll = roll - 90
                
                pts = compute_roi_rotation(roll, pt, 66).astype(int)
                y_new, x_new = pts[1], pts[0]
                print(f"Yaw: {yaw}, Roll: {roll}, Pitch: {pitch}")

                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break

                cv2.circle(frame, (x0, y0), 5, (0, 255, 0), -1)
                cv2.circle(frame, (x_new, y_new), 5, (0, 0, 255), -1)

                cv2.putText(frame, f"Roll: {roll:.2f}°", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"New Pixel: ({x_new}, {y_new})", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("Roll-based Pixel Transformation", frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    except Exception as e:
        print("Error:", e)

cap.release()
cv2.destroyAllWindows()
ser.close()
