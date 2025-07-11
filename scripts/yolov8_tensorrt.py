from ultralytics import YOLO

model = YOLO(r"D:\Aryabhatta\Yolov8_custom\scripts\models\best.pt")
model.export(
    format="engine")

# Load the exported TensorRT INT8 model
# model = YOLO("best.engine", task="detect")

# # Run inference
# result = model.predict(r"E:\Aryabhatta_motors_computer_vision\scripts\yolov8_test\Screenshot (9).png")