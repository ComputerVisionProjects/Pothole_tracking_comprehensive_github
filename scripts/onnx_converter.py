from ultralytics import YOLO

# Load a model
model = YOLO(r"yolov8n.pt")  # load an official model
model = YOLO(r"E:\Aryabhatta_motors_computer_vision\scripts\models\best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")