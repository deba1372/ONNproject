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
    # read with average for test_1 (trying to check whether averaging over RGB is useful or not)
    def __init__(self):
        """initialising the stuff"""
        self.all_points_array = np.array([])
        self.width = 0
        self.height = 0
        self.input = []
        self.biases = []
        self.weights = []

    def crop(self, image_path, coords, saved_location):
        """
        image_path: The path to the image to edit
        coords: A tuple of x/y coordinates (x1, y1, x2, y2)
        saved_location: Path to save the cropped image
        crops the image to the given coords and saves it
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
            if tup[0] > cutoff_value: # if the pixel is above a certain RGB value then its a bright pixel else dark
                tup1 = (255, 255, 255)
            else:
                tup1 = (0, 0, 0)
            output_list.append(tup1)
        return output_list

    def start(self):
        """ In order to initialise the self.all_points_array
        this method follows a general strategy of first having a grid in the center, then the last row and the last
        columns with alternating pixels lighting up and then finding the center points of those pixels and then
        interpolating to find the center of all pixels
        """
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
        # drawing a grid that is separated by 4 units in the vertical and 2 units in the horizontal
        for i in range(0, self.height, 4):
            for j in range(0, self.width, 2):
                draw.point([(j, i)], fill=255)  # x,y

        disp.image(image_1)
        disp.display()
        # made the display needed to find the points

        # Now get the pixel data

        camera = picamera.PiCamera()
        camera.resolution = (2592, 1944)
        camera.start_preview()
        camera.led = False
        time.sleep(2)
        camera.capture('full_multi_point_2.jpg')
        camera.stop_preview()   # taking a picture of the display
        image = 'full_multi_point_2.jpg'
        self.crop(image, (1020, 620, 1800, 1050), 'full_multi_crop_2.jpg')
        image = 'full_multi_crop_2.jpg'
        img = Image.open(image)
        pixels = list(img.getdata())
        pixels2 = []  # this part is kinda redundant now
        for pixel in pixels:
            total = 0
            for x in pixel:
                total += x
            total = int(total / 3)
            pixels2.append((total, total, total))

        filtered_list = self.filter(140, pixels2)   # cropped and filtered that image
        img = Image.new('RGB', Image.open('full_multi_crop_2.jpg').size)
        img.putdata(filtered_list)
        img.save('full_multi_filter_2.jpg')

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
            labelMask = np.zeros(gray.shape, dtype="uint8")
            labelMask[labels == label] = 255
            mask = cv2.add(mask, labelMask)

        pixel_point = []
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = contours.sort_contours(cnts)[0]
        # loop over the contours
        # pixel_point is going to contain the coords of all the bright pixels
        for (i, c) in enumerate(cnts):
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pixel_point.append((int(cY), int(cX)))
        # found the center points of the pixels

        # going to arrange all the bright pixels according to their vertical coords

        pixel_point_sort = sorted(pixel_point, key=lambda y: y[0])
        new_pixel_point = []
        for i in range(16):
            new_pixel_point.append(sorted(pixel_point_sort[64 * i:64 * (i + 1)], key=lambda z: z[1]))

        new_pixel_array = np.array([y for x in new_pixel_point for y in x])
        new_pixel_array.shape = (16, 64, 2)
        # new_pixel_array contains all the bright pixels in an arranged way

        self.all_points_array = np.array(np.zeros(64 * 128 * 2))
        self.all_points_array.shape = (64, 128, 2)
        for i in range(0, 16):
            for j in range(0, 64):
                self.all_points_array[60 - 4 * i][126 - 2 * j][0] = new_pixel_array[i][j][0]
                self.all_points_array[60 - 4 * i][126 - 2 * j][1] = new_pixel_array[i][j][1]
        # finding the coords of the bright pixels
        for i in range(0, 61, 4):
            for j in range(1, 127, 2):
                if j % 2 == 1:
                    self.all_points_array[i][j][0] = int((self.all_points_array[i][j - 1][0] + self.all_points_array[i][j + 1][0]) / 2)
                    self.all_points_array[i][j][1] = int((self.all_points_array[i][j - 1][1] + self.all_points_array[i][j + 1][1]) / 2)
        # interpolation
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

        # now drawing the last row
        i = 63
        for j in range(0, self.width, 2):
            draw.point([(j, i)], fill=255)  # x,y

        disp.image(image_1)
        disp.display()

        # Now get the pixel data

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
            labelMask = np.zeros(gray.shape, dtype="uint8")
            labelMask[labels == label] = 255
            mask = cv2.add(mask, labelMask)

        pixel_point = []
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = contours.sort_contours(cnts)[0]
        # loop over the contours
        for (i, c) in enumerate(cnts):
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pixel_point.append((int(cY), int(cX)))
        # found the center points

        pixel_point_sort = sorted(pixel_point, key=lambda z: z[1])
        # arranging and interpolation
        for j in range(64):
            self.all_points_array[63][126 - 2 * j][0] = pixel_point_sort[j][0]
            self.all_points_array[63][126 - 2 * j][1] = pixel_point_sort[j][1]
            self.all_points_array[62][126 - 2 * j][0] = (2*pixel_point_sort[j][0] + self.all_points_array[60][126 - 2 * j][0])/3
            self.all_points_array[62][126 - 2 * j][1] = (2*pixel_point_sort[j][1] + self.all_points_array[60][126 - 2 * j][1])/3
            self.all_points_array[61][126 - 2 * j][0] = (pixel_point_sort[j][0] + 2*self.all_points_array[60][126 - 2 * j][0])/3
            self.all_points_array[61][126 - 2 * j][1] = (pixel_point_sort[j][1] + 2*self.all_points_array[60][126 - 2 * j][1])/3

        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        for i in [61, 62, 63]:
            for j in range(126):
                if j % 2 == 1:
                    self.all_points_array[i][j][1] = (self.all_points_array[i][j-1][1] + self.all_points_array[i][j + 1][1])/ 2
                    self.all_points_array[i][j][0] = (self.all_points_array[i][j-1][0] + self.all_points_array[i][j+1][0])/2
        # now drawing the last column
        j = 127
        for i in range(0, self.width, 3):
            draw.point([(j, i)], fill=255)  # x,y

        disp.image(image_1)
        disp.display()

        # Now get the pixel data

        camera.start_preview()
        camera.led = False
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
            labelMask = np.zeros(gray.shape, dtype="uint8")
            labelMask[labels == label] = 255
            mask = cv2.add(mask, labelMask)

        pixel_point = []
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = contours.sort_contours(cnts)[0]
        # loop over the contours
        for (i, c) in enumerate(cnts):
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pixel_point.append((int(cY), int(cX)))
        # found the center points and then arranging and interpolating
        pixel_point_sort = sorted(pixel_point, key=lambda z: z[0])
        for j in range(64):
            if j % 3 == 0:
                self.all_points_array[j][127][0] = pixel_point_sort[int(21 - j/3)][0]
                self.all_points_array[j][127][1] = pixel_point_sort[int(21 - j/3)][1]
            elif j % 3 == 1:
                self.all_points_array[j][127][0] = (2*pixel_point_sort[21 - int((j-1)/3)][0] + pixel_point_sort[21 - int((j+2)/3)][0])/3
                self.all_points_array[j][127][1] = (2*pixel_point_sort[21 - int((j-1)/3)][1] + pixel_point_sort[21 - int((j+2)/3)][1])/3
            elif j % 3 == 2:
                self.all_points_array[j][127][0] = (2*pixel_point_sort[21 - int((j+1)/3)][0] + pixel_point_sort[21 - int((j-2)/3)][0])/3
                self.all_points_array[j][127][1] = (2*pixel_point_sort[21 - int((j+1)/3)][1] + pixel_point_sort[21 - int((j-2)/3)][1])/3

        img = Image.new('RGB', Image.open('last_col_filter.jpg').size)
        test_image_array = np.zeros(gray.shape)
        print(test_image_array.shape)
        for i in range(64):
            for j in range(128):
                test_image_array[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] = 1
        test_image_data = []
        for i in range(test_image_array.shape[0]):
            for j in range(test_image_array.shape[1]):
                if test_image_array[i][j] == 1:
                    test_image_data.append((255, 255, 255))
                else:
                    test_image_data.append((0, 0, 0))
        img.putdata(test_image_data)
        img.save('verify.jpg')
        # print(time.time() - start_time)
        camera.close()

    def test(self):
        """ testing whether the data input has been taken properly or not by assign 1 or 0 to each pixel random and
        then checking whether the computer can read the display accurately or not """
        # test_array contains the randomly selected led display data (rn its alternate pixels)
        test_array = np.array(np.zeros(64 * 128))
        test_array.shape = (64, 128)
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
            pixels2.append((pixel[2], pixel[2], pixel[2]))
        # using only blue pixels for reading
        filtered_list = self.filter(120, pixels2)
        img = Image.new('RGB', Image.open('test_crop.jpg').size)
        img.putdata(filtered_list)
        img.save('test_filter.jpg')
        img = cv2.imread('test_filter.jpg')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, None, iterations=1)

        # result_array would contains the data of the pixels that are being read
        result_array = np.array(np.zeros(64 * 128))
        result_array.shape = (64, 128)

        for i in range(64):
            for j in range(128):
                if gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] == 0:
                    result_array[i][j] = 0
                else:
                    result_array[i][j] = 1

        # checking whether result_array and test_array are in fact same or not
        errors_list = []
        errors = 0
        for i in range(0, 64, 4):
            for j in range(0, 128, 2):
                if (abs(result_array[i][j] - test_array[i][j])) > 0.1:
                    errors += 1
                    errors_list.append((i,j))

        print('errors', errors)
        print(len(errors_list))

        camera.close()

    def test_1(self, num):
        """ randomly selected as many points as the parameter num to be bright pixels and tests whether they can
        be identified or not """
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

        camera = picamera.PiCamera()
        camera.resolution = (2592, 1944)
        camera.start_preview()
        camera.led = False
        time.sleep(1)
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

        filtered_list = self.filter(150, pixels2)
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
        """ checks whether the display can be read or not when the led is on"""
        led = LED(21)
        camera = picamera.PiCamera()
        camera.resolution = (2592, 1944)
        camera.start_preview()
        led.on()

        test_list = []
        for r in range(num):
            test_list.append((int(np.random.randint(0, 64, size=1)/2)*2, int(np.random.randint(0, 128, size=1)/2)*2))
        # randomly lighting up a point

        RST = 24
        # 128x64 display with hardware I2C:
        disp = Adafruit_SSD1306.SSD1306_128_64(rst=RST)

        # Initialize library.
        disp.begin()

        # Clear display.
        disp.clear()
        disp.display()

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

        blue = []
        for point in pixels:
            blue.append((point[2], point[2], point[2]))
        img = Image.new('RGB', Image.open('led_crop.jpg').size)
        img.putdata(blue)
        img.save('led_crop_blue.jpg')

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

    def read_input_stream(self, input, weights):
        """this method is there to read the input and led display data passed onto the neuron
        input - whether led is on or off
        weights - led display data
        """
        camera = picamera.PiCamera()
        camera.resolution = (2592, 1944)
        led = LED(21)
        red_avg = []
        image = 'input.jpg'
        RST = 24
        disp = Adafruit_SSD1306.SSD1306_128_64(rst=RST)

        # Initialize library.
        disp.begin()

        # Clear display.
        disp.clear()
        disp.display()
        self.width = disp.width
        self.height = disp.height

        read_weights = np.array(np.zeros(len(input) * 64 * 128))
        read_weights.shape = (len(input), 64, 128)
        # done initialising
        for i in range(len(input)):
            if input[i] == 1:
                led.on()
            else:
                led.off()

            image_1 = Image.new('1', (self.width, self.height))
            # Get drawing object to draw on image.
            draw = ImageDraw.Draw(image_1)

            # Draw a black filled box to clear the image.
            draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

            for j in range(0, self.height):
                for k in range(0, self.width):
                    if int(weights[i][j][k]) == 1:
                        draw.point([(k, j)], fill=255)  # x,y

            disp.image(image_1)
            disp.display()
            # drawing and display done for one neuron
            print('cheese')
            camera.capture(image)
            # Now get the pixel data
            image_obj = Image.open(image)
            img = image_obj.crop((1020, 620, 1800, 1050))
            pixels = list(img.getdata())
            blue = []
            for point in pixels:
                blue.append((point[2], point[2], point[2]))
            # blue filter

            filtered_list = self.filter(120, blue)
            img = Image.new('RGB', Image.open(image).size)
            img.putdata(filtered_list)
            img.save('input_filter.jpg')
            img = cv2.imread('input_filter.jpg')

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.erode(gray, None, iterations=1)

            for j in range(64):
                for k in range(128):
                    if gray[int(self.all_points_array[j][k][0])][int(self.all_points_array[j][k][1])] == 0:
                        read_weights[i][j][k] = 0
                    else:
                        read_weights[i][j][k] = 1

        errors = 0

        for i in range(len(input)):
            for j in range(64):
                for k in range(128):
                    if abs(read_weights[i][j][k] - weights[i][j][k]) > .1:
                        errors += 1
        print('num of errors' + str(errors))
        camera.close()



if __name__ == '__main__':
    disp = Disp()
    disp.start()
    #for i in range(10):
    #    disp.test()
    for i in range(10):
        disp.test_1(4)
    # disp.read_with_led(1)

    #RST = 24
    # 128x64 display with hardware I2C:
    #disp_1 = Adafruit_SSD1306.SSD1306_128_64(rst=RST)

    #disp_1.begin()
    #disp_1.clear()
    #disp_1.display()

    #image_1 = Image.new('1', (disp_1.width, disp_1.height))

    # Get drawing object to draw on image.
    #draw = ImageDraw.Draw(image_1)

    # Draw a black filled box to clear the image.
    #draw.rectangle((0, 0, disp_1.width, disp_1.height), outline=0, fill=0)
    # drawing a grid that is separated by 4 units in the vertical and 2 units in the horizontal
  #  for i in range(0, disp_1.height, 4):
 #       for j in range(0, disp_1.width, 2):
#            draw.point([(j, i)], fill=255)  # x,y

   # disp_1.image(image_1)
    #disp_1.display()
    #weights = np.zeros(10*64*128)
    #weights.shape = (10, 64, 128)
    #input = []
    #for i in range(10):
     #   input.append(int(np.random.randint(0, 2, size=1)))
     #   for j in range(0, 64, 4):
      #      for k in range(0, 128, 2):
     #           weights[i][j][k] = 1 # int(np.random.randint(0, 2, size=1))

    #print('start')
    #start_time = time.time()
    #disp.read_input_stream(input, weights)
    #print('end_time = ' + str(time.time() - start_time))

