from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image
import time
import random
import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306
import time
import Image
import ImageDraw
import ImageFont
import picamera
import random
from gpiozero import LED


class Disp(object):

    def __init__(self):
        self.all_points_array = np.array([])
        self.width = 0
        self.height = 0

    def crop(self, image_path, coords, saved_location):
        """
        @param image_path: The path to the image to edit
        @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
        @param saved_location: Path to save the cropped image
        """
        image_obj = Image.open(image_path)
        cropped_image = image_obj.crop(coords)
        cropped_image.save(saved_location)

    def filter(self, cutoff_value, input_list):
        """takes in a list of tuples containing pixel data dn returns (255,255,255)
        for every pixel that is greater than cutoff and (0,0,0) for each pixel less than
        cutoff"""
        output_list = []
        for tup in input_list:
            if tup[0] > cutoff_value:
                tup1 = (255, 255, 255)
            else:
                tup1 = (0, 0, 0)
            output_list.append(tup1)
        return output_list

    def start(self):
        """ In order to initialise the self.all_points_array that is to find the center coords of all the pixels on the adafruit display """
        # Raspberry Pi pin configuration:
        RST = 24
        # 128x64 display with hardware I2C:
        disp = Adafruit_SSD1306.SSD1306_128_64(rst=RST)

        disp.begin()
        disp.clear()
        disp.display()

        self.width = disp.width
        self.height = disp.height
        image_1 = Image.new('1', (self.width, self.height))

        # Get drawing object to draw on image.
        draw = ImageDraw.Draw(image_1)

        # Draw a black filled box to clear the image.
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        # using only these pixels so that i could later sort as per vertical axis values and then horizontal
        for i in range(0, self.height, 4):
            for j in range(0, self.width, 2):
                draw.point([(j, i)], fill=255)  # x,y

        disp.image(image_1)
        disp.display()

        # Now get the pixel data

        camera = picamera.PiCamera()
        camera.resolution = (2592, 1944)
        camera.start_preview()
        camera.led = False
        time.sleep(2)
        camera.capture('full_multi_point_2.jpg')
        camera.stop_preview()
        image = 'full_multi_point_2.jpg'
        self.crop(image, (1020, 620, 1800, 1050), 'full_multi_crop_2.jpg')
        image = 'full_multi_crop_2.jpg'
        img = Image.open(image)
        pixels = list(img.getdata())
        pixels2 = []
        for pixel in pixels:
            total = 0
            for x in pixel:
                total += x
            total = int(total / 3)
            pixels2.append((total, total, total))
        # averaging over the RGB spectrum
        filtered_list = self.filter(140, pixels2)
        img = Image.new('RGB', Image.open('full_multi_crop_2.jpg').size)
        img.putdata(filtered_list)
        img.save('full_multi_filter_2.jpg')

        start_time = time.time()

        image = cv2.imread('full_multi_filter_2.jpg')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, None, iterations=1)

        new_image = Image.fromarray(gray)
        new_image.save('cv_proc_mult.png')

        labels = measure.label(gray, neighbors=8, background=0)
        mask = np.zeros(gray.shape, dtype="uint8")

        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue
            # otherwise, construct the label mask and count the
            # number of pixels
            labelMask = np.zeros(gray.shape, dtype="uint8")
            labelMask[labels == label] = 255

            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            mask = cv2.add(mask, labelMask)

        # find the contours in the mask, then sort them from left to
        # right

        pixel_point = []
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = contours.sort_contours(cnts)[0]
        # loop over the contours
        for (i, c) in enumerate(cnts):
            # draw the bright spot on the image
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pixel_point.append((int(cY), int(cX)))
            cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 0, 255), 2)

        # show the output image
        new_image = Image.fromarray(image)
        new_image.save('full_cv_proc_mult_2.jpg')

        # sorting the list as per vertical coordinate in order to get the rows
        pixel_point_sort = sorted(pixel_point, key=lambda y: y[0])
        new_pixel_point = []
        for i in range(16):
            new_pixel_point.append(sorted(pixel_point_sort[64 * i:64 * (i + 1)], key=lambda z: z[1]))
        # sorting all the rows as per the horizontal values
        new_pixel_array = np.array([y for x in new_pixel_point for y in x])
        new_pixel_array.shape = (16, 64, 2)

        # interpolating over the smaller regions
        self.all_points_array = np.array(np.zeros(64 * 128 * 2))
        self.all_points_array.shape = (64, 128, 2)
        for i in range(0, 16):
            for j in range(0, 64):
                self.all_points_array[60 - 4 * i][126 - 2 * j][0] = new_pixel_array[i][j][0]
                self.all_points_array[60 - 4 * i][126 - 2 * j][1] = new_pixel_array[i][j][1]

        for i in range(0, 61, 4):
            for j in range(1, 127, 2):
                self.all_points_array[i][j][0] = int((self.all_points_array[i][j - 1][0] + self.all_points_array[i][j + 1][0]) / 2)
                self.all_points_array[i][j][1] = int((self.all_points_array[i][j - 1][1] + self.all_points_array[i][j + 1][1]) / 2)

        for i in range(61):
            for j in range(127):
                if i % 4 == 1:
                    self.all_points_array[i][j][0] = int(
                        (3 * self.all_points_array[i - 1][j][0] + self.all_points_array[i + 3][j][0]) / 4)
                    self.all_points_array[i][j][1] = int(
                        (3 * self.all_points_array[i - 1][j][1] + self.all_points_array[i + 3][j][1]) / 4)
                elif i % 4 == 2:
                    self.all_points_array[i][j][0] = int(
                        (2 * self.all_points_array[i - 2][j][0] + 2 * self.all_points_array[i + 2][j][0]) / 4)
                    self.all_points_array[i][j][1] = int(
                        (2 * self.all_points_array[i - 2][j][1] + 2 * self.all_points_array[i + 2][j][1]) / 4)
                elif i % 4 == 3:
                    self.all_points_array[i][j][0] = int(
                        (self.all_points_array[i - 3][j][0] + 3 * self.all_points_array[i + 1][j][0]) / 4)
                    self.all_points_array[i][j][1] = int(
                        (self.all_points_array[i - 3][j][1] + 3 * self.all_points_array[i + 1][j][1]) / 4)

        # Draw a black filled box to clear the image.
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        # now to find the coords of the pixels on the last three rows by lighting up the
        # alternate points on the last row
        i = 63
        for j in range(0, self.width, 2):
            draw.point([(j, i)], fill=255)  # x,y

        disp.image(image_1)
        disp.display()

        camera.start_preview()
        camera.led = False
        time.sleep(2)
        camera.capture('last_row_disp.jpg')
        camera.stop_preview()

        image = 'last_row_disp.jpg'
        self.crop(image, (1020, 620, 1800, 1050), 'last_row_crop.jpg')

        image = 'last_row_crop.jpg'
        img = Image.open(image)
        pixels = list(img.getdata())
        pixels2 = []
        for pixel in pixels:
            total = 0
            for x in pixel:
                total += x
            total = int(total / 3)
            pixels2.append((total, total, total))

        filtered_list = self.filter(140, pixels2)
        img = Image.new('RGB', Image.open(image).size)
        img.putdata(filtered_list)
        img.save('last_row_filter.jpg')

        image = cv2.imread('last_row_filter.jpg')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, None, iterations=1)

        labels = measure.label(gray, neighbors=8, background=0)
        mask = np.zeros(gray.shape, dtype="uint8")

        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue

            # otherwise, construct the label mask and count the
            # number of pixels
            labelMask = np.zeros(gray.shape, dtype="uint8")
            labelMask[labels == label] = 255

            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            mask = cv2.add(mask, labelMask)

        # find the contours in the mask, then sort them from left to
        # right

        pixel_point = []
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = contours.sort_contours(cnts)[0]
        # loop over the contours
        for (i, c) in enumerate(cnts):
            # draw the bright spot on the image
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pixel_point.append((int(cY), int(cX)))
            cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 0, 255), 2)

        # show the output image
        pixel_point_sort = sorted(pixel_point, key=lambda z: z[1])

        # missed the filling in the middle parts between the alternate ones code - fixed it in later programs
        for j in range(64):
            self.all_points_array[63][126 - 2 * j][0] = pixel_point_sort[j][0]
            self.all_points_array[63][126 - 2 * j][1] = pixel_point_sort[j][1]
            self.all_points_array[62][126 - 2 * j][0] = (2*pixel_point_sort[j][0] + self.all_points_array[60][126 - 2 * j][0])/3
            self.all_points_array[62][126 - 2 * j][1] = (2*pixel_point_sort[j][1] + self.all_points_array[60][126 - 2 * j][1])/3
            self.all_points_array[61][126 - 2 * j][0] = (pixel_point_sort[j][0] + 2*self.all_points_array[60][126 - 2 * j][0])/3
            self.all_points_array[61][126 - 2 * j][1] = (pixel_point_sort[j][1] + 2*self.all_points_array[60][126 - 2 * j][1])/3

        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        j = 127
        for i in range(0, self.width, 3):
            draw.point([(j, i)], fill=255)  # x,y

        disp.image(image_1)
        disp.display()

        # Now get the pixel data

        camera.start_preview()
        time.sleep(2)
        camera.capture('last_col_disp.jpg')
        camera.stop_preview()
        image = 'last_col_disp.jpg'
        self.crop(image, (1020, 620, 1800, 1050), 'last_col_crop.jpg')
        image = 'last_col_crop.jpg'
        img = Image.open(image)
        pixels = list(img.getdata())
        pixels2 = []
        for pixel in pixels:
            total = 0
            for x in pixel:
                total += x
            total = int(total / 3)
            pixels2.append((total, total, total))

        filtered_list = self.filter(140, pixels2)
        img = Image.new('RGB', Image.open(image).size)
        img.putdata(filtered_list)
        img.save('last_col_filter.jpg')

        image = cv2.imread('last_col_filter.jpg')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, None, iterations=1)

        labels = measure.label(gray, neighbors=8, background=0)
        mask = np.zeros(gray.shape, dtype="uint8")

        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue

            # otherwise, construct the label mask and count the
            # number of pixels
            labelMask = np.zeros(gray.shape, dtype="uint8")
            labelMask[labels == label] = 255

            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            mask = cv2.add(mask, labelMask)

        # find the contours in the mask, then sort them from left to
        # right

        pixel_point = []
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = contours.sort_contours(cnts)[0]
        # loop over the contours
        for (i, c) in enumerate(cnts):
            # draw the bright spot on the image
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pixel_point.append((int(cY), int(cX)))
            cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 0, 255), 2)

        # show the output image

        pixel_point_sort = sorted(pixel_point, key=lambda z: z[0])
        for j in range(64):
            if j % 3 == 0:
                self.all_points_array[j][127][0] = pixel_point_sort[int(21 - j/3)][0]
                self.all_points_array[j][127][1] = pixel_point_sort[int(21 - j/3)][1]
            elif j % 3 == 1:
                self.all_points_array[j][127][0] = (2*pixel_point_sort[21 - int((j-1)/3)][0] + pixel_point_sort[21 - int((j+2)/3)][0])/3
                self.all_points_array[j][127][1] = (2*pixel_point_sort[21 - int((j-1)/3)][0] + pixel_point_sort[21 - int((j+2)/3)][1])/3
            elif j % 3 == 2:
                self.all_points_array[j][127][0] = (2*pixel_point_sort[21 - int((j+1)/3)][0] + pixel_point_sort[21 - int((j-2)/3)][0])/3
                self.all_points_array[j][127][1] = (2*pixel_point_sort[21 - int((j+1)/3)][0] + pixel_point_sort[21 - int((j-2)/3)][1])/3

        print(time.time() - start_time)
        camera.close()

    def test(self):
        # this method assigns a random array as a test array (right now, only alternating pixels are random)
        # and checks whether the program correctly reads it
        test_array = np.array(np.zeros(64, 128), dtype=int)
        for i in range(0, 64, 2):
            for j in range(0, 128, 2):
                test_array[i][j] = int(np.random.randint(0, 2, size=1))

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
        self.width = disp.width
        self.height = disp.height
        image_1 = Image.new('1', (self.width, self.height))

        # Get drawing object to draw on image.
        draw = ImageDraw.Draw(image_1)

        # Draw a black filled box to clear the image.
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        for i in range(0, self.height):
            for j in range(0, self.width):
                if int(test_array[i][j]) == 1:
                    draw.point([(j, i)], fill=255)  # x,y

        disp.image(image_1)
        disp.display()

        # Now get the pixel data

        camera = picamera.PiCamera()
        camera.resolution = (2592, 1944)
        camera.start_preview()
        camera.led = False
        time.sleep(2)
        camera.capture('test.jpg')
        camera.stop_preview()
        image = 'test.jpg'
        self.crop(image, (1020, 620, 1800, 1050), 'test_crop.jpg')
        image = 'test_crop.jpg'
        img = Image.open(image)
        pixels = list(img.getdata())
        pixels2 = []
        for pixel in pixels:
            total = 0
            for x in pixel:
                total += x
            total = int(total / 3)
            pixels2.append((total, total, total))

        filtered_list = self.filter(120, pixels2)
        img = Image.new('RGB', Image.open('test_crop.jpg').size)
        img.putdata(filtered_list)
        img.save('test_filter.jpg')
        img = cv2.imread('test_filter.jpg')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, None, iterations=1)

        result_array = np.array(np.zeros(64 , 128), dtype = int)

        for i in range(64):
            for j in range(128):
                if gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] == 0:
                    result_array[i][j] = 0
                else:
                    result_array[i][j] = 1

        errors_list = []
        errors = 0
        for i in range(0, 64, 4):
            for j in range(0, 128, 2):
                if (abs(result_array[i][j] - test_array[i][j])) > 0.1:
                    errors += 1
                    errors_list.append((i, j))

        print('errors', errors)
        print(len(errors_list))

        camera.close()

    def test_1(self, num):
        """ this method randomly chooses as many points as num and makes them bright pixels and
        checks whether the program reads  them correctly
        """
        test_list = []
        for r in range(num):
            test_list.append((int(np.random.randint(0, 64, size=1)), int(np.random.randint(0, 128, size=1))))

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
        self.width = disp.width
        self.height = disp.height
        image_1 = Image.new('1', (self.width, self.height))

        # Get drawing object to draw on image.
        draw = ImageDraw.Draw(image_1)

        # Draw a black filled box to clear the image.
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        for point in test_list:
            draw.point([(point[1], point[0])], fill=255)

        disp.image(image_1)
        disp.display()

        # Now get the pixel data

        camera = picamera.PiCamera()
        camera.resolution = (2592, 1944)
        camera.start_preview()
        camera.led = False
        time.sleep(2)
        camera.capture('test.jpg')
        camera.stop_preview()
        image = 'test.jpg'
        self.crop(image, (1020, 620, 1800, 1050), 'test_crop.jpg')
        image = 'test_crop.jpg'
        img = Image.open(image)
        pixels = list(img.getdata())
        pixels2 = []
        for pixel in pixels:
            total = 0
            for x in pixel:
                total += x
            total = int(total / 3)
            pixels2.append((total, total, total))

        filtered_list = self.filter(120, pixels2)
        img = Image.new('RGB', Image.open('test_crop.jpg').size)
        img.putdata(filtered_list)
        img.save('test_filter.jpg')
        img = cv2.imread('test_filter.jpg')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, None, iterations=1)
        result_list = []
        for i in range(64):
            for j in range(128):
                if not (gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] == 0):
                    result_list.append((i, j))

        print('test_value', test_list,'result_value', result_list)
        camera.close()

    def read_with_led(self, num):
        """ randomly selects as many even numbered points as num and checks whether the program can read the points
        when the led is on """
        led = LED(21)
        camera = picamera.PiCamera()
        camera.resolution = (2592, 1944)
        camera.start_preview()
        led.on()
        test_list = []

        for r in range(num):
            test_list.append((int(np.random.randint(0, 64, size=1)/2)*2, int(np.random.randint(0, 128, size=1)/2)*2))

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
        self.width = disp.width
        self.height = disp.height
        image_1 = Image.new('1', (self.width, self.height))

        # Get drawing object to draw on image.
        draw = ImageDraw.Draw(image_1)

        # Draw a black filled box to clear the image.
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        for point in test_list:
            draw.point([(point[1], point[0])], fill=255)

        disp.image(image_1)
        disp.display()

        # Now get the pixel data
        image = 'led.jpg'
        camera.capture(image)
        camera.stop_preview()

        self.crop(image, (1020, 620, 1800, 1050), 'led_crop.jpg')
        image = 'led_crop.jpg'
        img = Image.open(image)
        pixels = list(img.getdata())
        red = []
        for point in pixels:
            red.append((point[0], point[0], point[0]))
        img = Image.new('RGB', Image.open('led_crop.jpg').size)
        img.putdata(red)
        img.save('led_crop_red.jpg')
        # led_crop_red is there in order to find whether the signal is 1 or 0
        blue = []
        for point in pixels:
            blue.append((point[2], point[2], point[2]))
        img = Image.new('RGB', Image.open('led_crop.jpg').size)
        img.putdata(blue)
        img.save('led_crop_blue.jpg')
        # using blue for reading the array
        image = 'led_crop_blue.jpg'
        img = Image.open(image)
        pixels = list(img.getdata())

        filtered_list = self.filter(120, pixels)
        img = Image.new('RGB', Image.open(image).size)
        img.putdata(filtered_list)
        img.save('led_blue_filter.jpg')
        img = cv2.imread('led_blue_filter.jpg')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, None, iterations=1)
        result_list = []
        for i in range(64):
            for j in range(128):
                if not (gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] == 0):
                    result_list.append((i, j))

        print('test_value', test_list,'result_value', result_list)
        camera.close()


if __name__ == '__main__':
    disp = Disp()
    disp.start()
    # for i in range(10):
    #    disp.test()
    # for i in range(10):
    #    disp.test_1(4)
    disp.read_with_led(1)
