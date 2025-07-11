# Script for exporting the model into various formats.

The following section outlines the code used to export the trained model into various formats. By default, training a YOLO model produces a `.pt` file, which is the native PyTorch format.

## Script for converting the model to the ONNX format.

``` py title="onnx_converter.py" linenums="1"
from ultralytics import YOLO

# Load a model
model = YOLO(r"yolov8n.pt")  # load an official model
model = YOLO(r"E:\Aryabhatta_motors_computer_vision\scripts\models\best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")
```

This code uses the native yolo library called ultralytics to export the yolo model into onnx.

## Code used to export the yolo model to openvino

```py title="yolo_openvino_export.py" linenums="1"

from ultralytics import YOLO  # For YOLOv5 and YOLOv8

model = YOLO(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\best(1).pt")

model.export(format="openvino")

```

The above script is used to convert the YOLO model into the OpenVINO format, which is a quantized version of the model.

## Script for exporting the model to the TensorRT format.

TensorRT is an optimization library designed to enhance inference speed. While it offers several other benefits, I won't go into those details here.

```py title="yolov8_tensorrt.py" linenums="1"

from ultralytics import YOLO

model = YOLO(r"D:\Aryabhatta\Yolov8_custom\scripts\models\best.pt")
model.export(
    format="engine")
```

