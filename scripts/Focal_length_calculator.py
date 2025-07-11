KNOWN_DISTANCE = 50
KNOWN_WIDTH = 14.5
face_width_in_frame = 238

def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image* measured_distance)/ real_width
    return focal_length

def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length)/face_width_in_frame
    return distance

focal_length = FocalLength(KNOWN_DISTANCE, KNOWN_WIDTH, face_width_in_frame)

distance = Distance_finder(focal_length, KNOWN_WIDTH, face_width_in_frame)

print(focal_length, distance)



