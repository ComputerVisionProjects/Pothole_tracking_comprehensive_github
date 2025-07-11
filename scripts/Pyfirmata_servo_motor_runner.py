import pyfirmata
from pyfirmata import Arduino, SERVO, util
from time import sleep
port = 'COM7' #usb pin
board = pyfirmata.Arduino(port)
pin = 6

def rotate_servo(angle):
     board.digital[pin].mode = SERVO
     board.digital[pin].write(angle)
     sleep(0.0015)

