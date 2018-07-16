import time

import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306

from PIL import Image
import ImageDraw
import ImageFont

import matplotlib.pyplot as py
import numpy as np
import picamera

# ignore - very ineffective method

def mean_bright_pt(pixels_array, start_height, end_height, start_width, end_width):
    '''returns the coordinates of a bright_pt'''
    num_pts = 0
    total_height = 0
    total_width = 0
    for i in range(start_height, end_height):
        for j in range(start_width, end_width):
            if pixels_array[i][j][0] == 255:
                weight = 1
            else:
                weight = 0
            num_pts += weight
            total_height += i * weight
            total_width += j * weight
    mean_height = int(total_height / num_pts)
    mean_width = int(total_width / num_pts)
    return ((mean_height, mean_width))

def filter(cutoff_value , input_list):
    """takes in a list of tuples containing pixel data dn returns (255,255,255)
    for every pixel that is greater than cutoff and (0,0,0) for each pixel less than
    cutoff"""
    output_list = []
    for tup in input_list:
        if tup[0] > cutoff_value:
            tup1 = (255,255,255)
        else:
            tup1 = (0,0,0)
        output_list.append(tup1)
    return output_list

def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.show()

if __name__ == '__main__':
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
    point_image = 'point.jpg'
    crop_image = 'cropped-point.jpg'
    filter_image = 'filtered-point.jpg'

    b = np.zeros(width)
    point_data = []
    for i in range(height):
        point_data.append(list(b))

    for i in range(height):
        for j in range(width):

    # Draw a black filled box to clear the image.
            draw.rectangle((0, 0, width, height), outline=0, fill=0)
            draw.point([(i, j)], fill = 255)
            disp.image(image)
            disp.display()

            # Now get the pixel data
            camera = picamera.PiCamera()
            camera.resolution = (2592,1944)
            camera.start_preview()
            camera.capture(point_image)
            camera.stop_preview()

            crop(point_image, (1050, 670, 1900, 1100), crop_image)
            img = Image.open(crop_image)
            pixels = list(img.getdata())
            pixels2 = []
            for pixel in pixels:
                total = 0
                for x in pixel:
                    total += x
                total = int(total/3)
                pixels2.append((total,total,total))

            filtered_list = filter(100, pixels2)
            img = Image.new('RGB' , Image.open(crop_image).size)
            img.putdata(filtered_list)
            img.show()
            img.save(filter_image)


            img = Image.open(filter_image)
            pixels = list(img.getdata())
            pixels_array = py.imread(filter_image)
            height, width, depth = pixels_array.shape

            point_data[i][j] = mean_bright_pt(pixels_array, 0, height , 0, width)

    print(type(point_data))
    print(len(point_data))
    print(type(point_data[0]))
    print(type(point_data[0][0]))