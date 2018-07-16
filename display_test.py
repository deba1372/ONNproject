
import time

import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306

from PIL import Image
import ImageDraw

import matplotlib.pyplot as py
import numpy as np
import picamera

if __name__ == '__main__':
    '''in order to test the display speed of the adafruit display '''

    # Raspberry Pi pin configuration:
    RST = 24
    start_time = time.time()
    # 128x64 display with hardware I2C:
    disp = Adafruit_SSD1306.SSD1306_128_64(rst=RST)
    # Initialize library.
    disp.begin()
    # Clear display.
    disp.clear()
    disp.display()
    init_time = time.time() - start_time
    width = disp.width
    height = disp.height
    image = Image.new('1', (width, height))
    camera = picamera.PiCamera()

    draw = ImageDraw.Draw(image)
    start_time_2 = time.time()

    just_disp_time = 0
    for i in range(5):
        for j in range(5):
            draw.rectangle((0, 0, width, height), outline=0, fill=0)
            draw.point([(j, i)], fill=255)
            disp.image(image)
            start_time_3 = time.time()
            disp.display()   # this line takes .128 secs
            end_time_3 = time.time()
            just_disp_time += (end_time_3 - start_time_3)/25
    disp_time = (time.time()- start_time_2)/25
    print('init time', init_time)
    print('disp time', disp_time)
    print('just disp time', just_disp_time)
