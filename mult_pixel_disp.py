import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306
import time
import Image
import ImageDraw
import ImageFont

import picamera

# to get the display with an eighth of the pixels lit up

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


for i in range(0, height, 4):
    for j in range(0, width, 2):
        draw.point([(j, i)], fill=255)  # x,y

disp.image(image)
disp.display()

# Now get the pixel data

camera = picamera.PiCamera()
camera.resolution = (2592, 1944)
camera.start_preview()
camera.led = False
time.sleep(2)
camera.capture('full_multi_point_2.jpg')
camera.stop_preview()
