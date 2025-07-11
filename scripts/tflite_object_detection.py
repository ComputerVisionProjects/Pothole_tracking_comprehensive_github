import numpy as np
import tensorflow as tf
import cv2
import time
print(tf.__version__)

Model_Path = r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\best3.tflite"
Video_path = r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\videos\WhatsApp Video 2024-07-05 at 01.41.50_72d4a5c5.mp4"

interpreter = tf.lite.Interpreter(model_path=Model_Path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
class_names = ['pothole']

cap = cv2.VideoCapture(Video_path)
ok, frame_image = cap.read()
original_image_height, original_image_width, _ = frame_image.shape
thickness = original_image_height // 500  
fontsize = original_image_height / 1500
print(thickness)
print(fontsize)

import cv2
import numpy as np

def preprocess_image(image, target_shape):
    """
    Preprocess the image to match the model's input requirements.
    - Resize to target_shape
    - Normalize to [0, 1]
    - Convert to FLOAT32
    """
    # Resize image to the required dimensions (256x256)
    image_resized = cv2.resize(image, (target_shape[1], target_shape[2]))

    # Normalize the image to the range [0, 1]
    image_resized = image_resized.astype(np.float32) / 255.0

    # Add a batch dimension (model expects a batch of images)
    input_data = np.expand_dims(image_resized, axis=0)

    return input_data

while True:
    ok, frame_image = cap.read()
    if not ok:
        break

    model_interpreter_start_time = time.time()
    # resize_img = cv2.resize(frame_image, (300, 300), interpolation=cv2.INTER_CUBIC)
    # reshape_image = resize_img.reshape(300, 300, 3)
    # image_np_expanded = np.expand_dims(reshape_image, axis=0)
    image_np_expanded = preprocess_image(frame_image, input_details[0]['shape'])

    interpreter.set_tensor(input_details[0]['index'], image_np_expanded) 
    interpreter.invoke()
    print(output_details)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data_1 = interpreter.get_tensor(output_details[1]['index']) 
    output_data_2 = interpreter.get_tensor(output_details[2]['index'])
    output_data_3 = interpreter.get_tensor(output_details[3]['index'])  
    each_interpreter_time = time.time() - model_interpreter_start_time

    for i in range(len(output_data_1[0])):
        confidence_threshold = output_data_2[0][i]
        if confidence_threshold > 0.3:
            label = "{}: {:.2f}% ".format(class_names[int(output_data_1[0][i])], output_data_2[0][i] * 100) 
            label2 = "inference time : {:.3f}s" .format(each_interpreter_time)
            left_up_corner = (int(output_data[0][i][1]*original_image_width), int(output_data[0][i][0]*original_image_height))
            left_up_corner_higher = (int(output_data[0][i][1]*original_image_width), int(output_data[0][i][0]*original_image_height)-20)
            right_down_corner = (int(output_data[0][i][3]*original_image_width), int(output_data[0][i][2]*original_image_height))
            cv2.rectangle(frame_image, left_up_corner_higher, right_down_corner, (0, 255, 0), thickness)
            cv2.putText(frame_image, label, left_up_corner_higher, cv2.FONT_HERSHEY_DUPLEX, fontsize, (255, 255, 255), thickness=thickness)
            cv2.putText(frame_image, label2, (30, 30), cv2.FONT_HERSHEY_DUPLEX, fontsize, (255, 255, 255), thickness=thickness)
    cv2.namedWindow('detect_result', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('detect_result', 800, 600)
    cv2.imshow("detect_result", frame_image)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
    elif key == 32:
        cv2.waitKey(0)
        continue
cap.release()
cv2.destroyAllWindows()