import pyfirmata
from pyfirmata import Arduino, SERVO, util
from time import sleep
port = 'COM7' #usb pin
pin = 10 #pin which servo is connected to on digital
pin1 = 9
board = pyfirmata.Arduino(port)
board.digital[pin].mode = SERVO
board.digital[pin1].mode = SERVO    

def rotate_servo(pin,angle):
     board.digital[pin].write(angle)
     board.digital[pin1].write(angle)
     sleep(0.0015)


user_angle_1 = int(input("Input user angle1: "))
user_angle_2 = int(input("Input user angle2: "))

rotate_servo(pin, user_angle_1)
rotate_servo(pin1, user_angle_2)