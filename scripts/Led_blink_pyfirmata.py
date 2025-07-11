import pyfirmata
import time

# Replace 'COM3' with the appropriate port for your Arduino
board = pyfirmata.Arduino('COM7') 

# Set up pin 13 as an output
led_pin = board.get_pin('d:13:o')

while True:
    led_pin.write(1)  # Turn LED on
    time.sleep(1)     # Wait for 1 second
    led_pin.write(0)  # Turn LED off
    time.sleep(1)