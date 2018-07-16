import time

import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306

from PIL import Image
import ImageDraw

import matplotlib.pyplot as py
import numpy as np
import picamera

# ignore - very ineffective method to find centers of points


def mean_bright_pt(input_list, start_height, end_height, start_width, end_width, cutoff_value):
    """returns the coordinates of the center of a  bright_pt
    input_list: the list of pixel data
    the rest specify the region in which the bright point is located
    """
    pixels_array = np.array(input_list)
    pixels_array.shape = (430, 850, 3)
    num_pts = 0
    total_height = 0
    total_width = 0
    # takes the array and finds the mean of all the pixels that are above the cutoff
    for i in range(start_height, end_height):
        for j in range(start_width, end_width):
            if (pixels_array[i][j][0] + pixels_array[i][j][1] + pixels_array[i][j][2])> 3*cutoff_value:
                weight = 1
            else:
                weight = 0
            num_pts += weight
            total_height += i * weight
            total_width += j * weight
    mean_height = int(total_height / num_pts)
    mean_width = int(total_width / num_pts)
    return (mean_height, mean_width)


def crop(image_path, coords):
    """
    takes an image as input, crops it to the coordinates provided and returns the pixel data
    of the cropped image as output
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)

    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    return list(cropped_image.getdata())


if __name__ == '__main__' :

    # Raspberry Pi pin configuration:
    RST = 24

    start_time = time.time()
    start_time_2 = time.time()
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
    camera = picamera.PiCamera()

    draw = ImageDraw.Draw(image)
    point_image = 'point.jpg'

    b = list(np.zeros(width))
    point_data = []
    for i in range(height):
        point_data.append(list(b))

    point_data_list = []

    init_time = time.time() - start_time_2

    disp_time = 0
    crop_time = 0
    data_input_time = 0
    pic_time = 0

    for i in range(5):
        print('abc')
        for j in range(5):

            # Draw a black filled box to clear the image.
            start_time_3 = time.time()

            draw.rectangle((0, 0, width, height), outline=0, fill=0)
            draw.point([(j, i)], fill=255)
            disp.image(image)
            disp.display()

            disp_time += (time.time() - start_time_3) / 25
            # Now get the pixel data

            start_time_6 = time.time()
            camera.resolution = (2592,1944)
            camera.capture(point_image)
            pic_time += (time.time() - start_time_6)/25

            start_time_5 = time.time()

            pixels = crop(point_image, (1050, 670, 1900, 1100))

            crop_time += (time.time() - start_time_5)/25

            start_time_4 = time.time()

            if j == 0:
                #  point_data[i][j] = mean_bright_pt(pixels, 0, 430, 0, 850, 120)
                a, b = mean_bright_pt(pixels, 0, 430, 0, 850, 120)
            else:
                # point_data[i][j] = mean_bright_pt(pixels, a - 10, a + 10, b - 20, b,120)
                # (a, b) = point_data[i][j]
                a, b = mean_bright_pt(pixels, a - 10, a + 10, b - 20, b, 120)
            point_data_list.append((a, b))
            data_input_time += (time.time() - start_time_4)/25

    print(type(point_data))
    print(len(point_data))
    print(type(point_data[0]))
    print(type(point_data[0][0]))
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time/25

    print ("total avg time", avg_time)
    print('crop time', crop_time)
    print('display time', disp_time)
    print('data input time', data_input_time)
    print('pic time', pic_time)
    print('init time', init_time)

    #print(point_data[1:3][1:3])