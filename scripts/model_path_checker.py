import os

file_path = r"D:\Arybhatta_motors_computer_vision\Yolov8_custom\scripts\models\best-yolov11n.pt"
if os.path.exists(file_path):
    print("File exists!")
else:
    print("File does not exist.")
