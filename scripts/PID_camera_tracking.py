import cv2
import numpy as np
import time
import serial
from simple_pid import PID
from ultralytics import YOLO  # Ensure you have ultralytics installed
import torch



torch.cuda.set_device(0)
# Load YOLO model
model = YOLO(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\best-yolov11n.pt")  # Change this to your YOLO model
print("before: ",model.device.type)
results = model(r"E:\Aryabhatta_motors_computer_vision\images_potholes\78778.png")
print("after: ",model.device.type)
# Arduino setup (Change COM7 to your port)
arduino = serial.Serial(port='COM7', baudrate=9600, timeout=1)
time.sleep(2)  # Wait for the connection

# PID controllers for two motors
pid_x = PID(0.02, 0, 0, setpoint=0)
pid_y = PID(0.02, 0, 0, setpoint=0)

pid_x.output_limits = (-45, 45)  # Servo range
pid_y.output_limits = (-45, 45)

# Deadband range (tolerance)
deadband_x = 10 # Pixels (horizontal tolerance)
deadband_y = 15 # Pixels (vertical tolerance)

# Camera setup
cap = cv2.VideoCapture(1)

# Initial servo positions
servo_angle_x = 90 # Start in center
servo_angle_y = 40

def send_servo_command(servo, angle):
    """ Send servo movement command to Arduino via Serial """
    command = f"{servo}:{int(angle)}\n"
    arduino.write(command.encode())  # Send command as bytes
    time.sleep(0.02)  # Small delay for stability

# Set initial servo positions
send_servo_command("X", servo_angle_x)
send_servo_command("Y", servo_angle_y)

try:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        results = model.track(frame, persist = True, conf = 0.5)

        color = (255, 248, 150) 

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates in format [x1, y1, x2, y2]
                object_center_x = (x1 + x2)/2
                object_center_y = (y1 + y2)/2
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2

                # Compute errors
                error_x = object_center_x - frame_center_x
                error_y = object_center_y - frame_center_y

                # Apply deadband logic: No correction if within range
                if abs(error_x) > deadband_x:
                    error_x = error_x.cpu().numpy()
                    correction_x = pid_x(error_x)
                    servo_angle_x = np.clip(servo_angle_x - correction_x, 0, 180)
                    print(servo_angle_x)        
                    send_servo_command("X", servo_angle_x)

                if abs(error_y) > deadband_y:
                    error_y = error_y.cpu().numpy()
                    correction_y = pid_y(error_y)
                    servo_angle_y = np.clip(servo_angle_y - correction_y, 0, 60)
                    print(servo_angle_y)
                    send_servo_command("Y", servo_angle_y)

        
        annotated_frame = results[0].plot()        
        cv2.imshow("Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
