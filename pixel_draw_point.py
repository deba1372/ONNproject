import time

import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306

import Image
import ImageDraw
import ImageFont

import picamera

# Will be getting a display here

# Raspberry Pi pin configuration:
RST = 24


# 128x64 display with hardware I2C:
disp = Adafruit_SSD1306.SSD1306_128_64(rst=RST)

# Initialize library.
disp.begin()

# Clear display.
disp.clear()
disp.display()

# Create blank image for drawing.
# Make sure to create image with mode '1' for 1-bit color.
width = disp.width
height = disp.height
image = Image.new('1', (width, height))

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)

# Draw a black filled box to clear the image.
draw.rectangle((0, 0, width, height), outline=0, fill=0)
draw.point([(0, 0)] , fill=255)

disp.image(image)
disp.display()

# Now get the pixel data

camera = picamera.PiCamera()
camera.resolution = (2592, 1944)
camera.start_preview()
camera.led = False
time.sleep(2)
camera.capture('point.jpg')
camera.stop_preview()
