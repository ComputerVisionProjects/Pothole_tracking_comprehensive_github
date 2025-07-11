from ultralytics import YOLO  # For YOLOv5 and YOLOv8

model = YOLO(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\best(1).pt")

model.export(format="openvino")