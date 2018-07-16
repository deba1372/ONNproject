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
    # reading done with blue. attempting to use just 1 or 2 threshold values to be applied
    # all over the image in order to do the data reading

    def __init__(self):
        """initialising the stuff
        all_points_array: contains the center points of all pixels on the led display
        width: width of display
        height: height of display
        input: the input vector to the net
        biases: contains the biases of all the neurons
        weights: contains the weights of all the neurons
        threshold_array: an array of threshold for the led array - not really used in this one
                        but refer to test_neural_network_input_2
        """
        self.all_points_array = np.array([])
        self.all_points_array_with_led = np.array([])
        self.width = 0
        self.height = 0
        self.input = []
        self.biases = []
        self.weights = []
        self.threshold_array = np.array([])

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
            if tup[0] + tup[1] + tup[2] > 3*cutoff_value: # if the pixel is above a certain RGB value then its a bright pixel else dark
                tup1 = (255, 255, 255)
            else:
                tup1 = (0, 0, 0)
            output_list.append(tup1)
        return output_list

    def start(self):
        """ In order to initialise the self.all_points_array """
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
        # pixel_point is going to contain the coords of all the bright pixels
        for (i, c) in enumerate(cnts):
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pixel_point.append((int(cY), int(cX)))

        # labeling all the bright points

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
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pixel_point.append((int(cY), int(cX)))

        pixel_point_sort = sorted(pixel_point, key=lambda z: z[1])

        for j in range(64):
            self.all_points_array[63][126 - 2 * j][0] = pixel_point_sort[j][0]
            self.all_points_array[63][126 - 2 * j][1] = pixel_point_sort[j][1]
            self.all_points_array[62][126 - 2 * j][0] = (2*pixel_point_sort[j][0] + self.all_points_array[60][126 - 2 * j][0])/3
            self.all_points_array[62][126 - 2 * j][1] = (2*pixel_point_sort[j][1] + self.all_points_array[60][126 - 2 * j][1])/3
            self.all_points_array[61][126 - 2 * j][0] = (pixel_point_sort[j][0] + 2*self.all_points_array[60][126 - 2 * j][0])/3
            self.all_points_array[61][126 - 2 * j][1] = (pixel_point_sort[j][1] + 2*self.all_points_array[60][126 - 2 * j][1])/3
        # interpolation
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        for i in [61,62,63]:
            for j in range(126):
                if j%2 == 1:
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
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pixel_point.append((int(cY), int(cX)))

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
        img.save('verify.jpg')  # can be used to visualize whether or not this above steps are done correctly
        # print(time.time() - start_time)
        camera.close()

    # some of the older programs forgot about interpolating on the top row


    def start_with_led(self):
        """ In order to initialise the self.all_points_array but with led on - just to check """
        # Raspberry Pi pin configuration:
        led = LED(21)
        led.on()
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

        camera = picamera.PiCamera()
        camera.resolution = (2592, 1944)
        camera.start_preview()
        camera.led = False
        time.sleep(2)
        camera.capture('full_multi_point_2.jpg')
        camera.stop_preview()  # taking a picture of the display
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

        filtered_list = self.filter(125, pixels2)  # cropped and filtered that image
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
        # pixel_point is going to contain the coords of all the bright pixels
        for (i, c) in enumerate(cnts):
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pixel_point.append((int(cY), int(cX)))

        # going to arrange all the bright pixels according to their vertical coords

        pixel_point_sort = sorted(pixel_point, key=lambda y: y[0])
        new_pixel_point = []
        for i in range(16):
            new_pixel_point.append(sorted(pixel_point_sort[64 * i:64 * (i + 1)], key=lambda z: z[1]))

        new_pixel_array = np.array([y for x in new_pixel_point for y in x])
        new_pixel_array.shape = (16, 64, 2)
        # new_pixel_array contains all the bright pixels in an arranged way

        self.all_points_array_with_led = np.array(np.zeros(64 * 128 * 2))
        self.all_points_array_with_led.shape = (64, 128, 2)
        for i in range(0, 16):
            for j in range(0, 64):
                self.all_points_array_with_led[60 - 4 * i][126 - 2 * j][0] = new_pixel_array[i][j][0]
                self.all_points_array_with_led[60 - 4 * i][126 - 2 * j][1] = new_pixel_array[i][j][1]
        # finding the coords of the bright pixels
        for i in range(0, 61, 4):
            for j in range(1, 127, 2):
                if j % 2 == 1:
                    self.all_points_array_with_led[i][j][0] = int(
                        (self.all_points_array_with_led[i][j - 1][0] + self.all_points_array_with_led[i][j + 1][0]) / 2)
                    self.all_points_array_with_led[i][j][1] = int(
                        (self.all_points_array_with_led[i][j - 1][1] + self.all_points_array_with_led[i][j + 1][1]) / 2)
        # interpolation
        for i in range(61):
            for j in range(127):
                if i % 4 == 1:
                    self.all_points_array_with_led[i][j][0] = int(
                        (3 * self.all_points_array_with_led[i - 1][j][0] + self.all_points_array_with_led[i + 3][j][0]) / 4)
                    self.all_points_array_with_led[i][j][1] = int(
                        (3 * self.all_points_array_with_led[i - 1][j][1] + self.all_points_array_with_led[i + 3][j][1]) / 4)
                elif i % 4 == 2:
                    self.all_points_array_with_led[i][j][0] = int(
                        (2 * self.all_points_array_with_led[i - 2][j][0] + 2 * self.all_points_array_with_led[i + 2][j][0]) / 4)
                    self.all_points_array_with_led[i][j][1] = int(
                        (2 * self.all_points_array_with_led[i - 2][j][1] + 2 * self.all_points_array_with_led[i + 2][j][1]) / 4)
                elif i % 4 == 3:
                    self.all_points_array_with_led[i][j][0] = int(
                        (self.all_points_array_with_led[i - 3][j][0] + 3 * self.all_points_array_with_led[i + 1][j][0]) / 4)
                    self.all_points_array_with_led[i][j][1] = int(
                        (self.all_points_array_with_led[i - 3][j][1] + 3 * self.all_points_array_with_led[i + 1][j][1]) / 4)

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

        filtered_list = self.filter(130, pixels2)
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
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pixel_point.append((int(cY), int(cX)))

        pixel_point_sort = sorted(pixel_point, key=lambda z: z[1])
        print(len(pixel_point_sort))
        for j in range(64):
            self.all_points_array_with_led[63][126 - 2 * j][0] = pixel_point_sort[j][0]
            self.all_points_array_with_led[63][126 - 2 * j][1] = pixel_point_sort[j][1]
            self.all_points_array_with_led[62][126 - 2 * j][0] = (2 * pixel_point_sort[j][0] +
                                                         self.all_points_array_with_led[60][126 - 2 * j][0]) / 3
            self.all_points_array_with_led[62][126 - 2 * j][1] = (2 * pixel_point_sort[j][1] +
                                                         self.all_points_array_with_led[60][126 - 2 * j][1]) / 3
            self.all_points_array_with_led[61][126 - 2 * j][0] = (pixel_point_sort[j][0] + 2 *
                                                         self.all_points_array_with_led[60][126 - 2 * j][0]) / 3
            self.all_points_array_with_led[61][126 - 2 * j][1] = (pixel_point_sort[j][1] + 2 *
                                                         self.all_points_array_with_led[60][126 - 2 * j][1]) / 3

        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        for i in [61, 62, 63]:
            for j in range(126):
                if j % 2 == 1:
                    self.all_points_array_with_led[i][j][1] = (self.all_points_array_with_led[i][j - 1][1] + self.all_points_array_with_led[i][j + 1][
                        1]) / 2
                    self.all_points_array_with_led[i][j][0] = (self.all_points_array_with_led[i][j - 1][0] + self.all_points_array_with_led[i][j + 1][
                        0]) / 2
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
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pixel_point.append((int(cY), int(cX)))

        pixel_point_sort = sorted(pixel_point, key=lambda z: z[0])
        for j in range(64):
            if j % 3 == 0:
                self.all_points_array_with_led[j][127][0] = pixel_point_sort[int(21 - j / 3)][0]
                self.all_points_array_with_led[j][127][1] = pixel_point_sort[int(21 - j / 3)][1]
            elif j % 3 == 1:
                self.all_points_array_with_led[j][127][0] = (2 * pixel_point_sort[21 - int((j - 1) / 3)][0] +
                                                    pixel_point_sort[21 - int((j + 2) / 3)][0]) / 3
                self.all_points_array_with_led[j][127][1] = (2 * pixel_point_sort[21 - int((j - 1) / 3)][1] +
                                                    pixel_point_sort[21 - int((j + 2) / 3)][1]) / 3
            elif j % 3 == 2:
                self.all_points_array_with_led[j][127][0] = (2 * pixel_point_sort[21 - int((j + 1) / 3)][0] +
                                                    pixel_point_sort[21 - int((j - 2) / 3)][0]) / 3
                self.all_points_array_with_led[j][127][1] = (2 * pixel_point_sort[21 - int((j + 1) / 3)][1] +
                                                    pixel_point_sort[21 - int((j - 2) / 3)][1]) / 3

        img = Image.new('RGB', Image.open('last_col_filter.jpg').size)
        test_image_array = np.zeros(gray.shape)
        print(test_image_array.shape)
        for i in range(64):
            for j in range(128):
                test_image_array[int(self.all_points_array_with_led[i][j][0])][int(self.all_points_array_with_led[i][j][1])] = 1
        test_image_data = []
        for i in range(test_image_array.shape[0]):
            for j in range(test_image_array.shape[1]):
                if test_image_array[i][j] == 1:
                    test_image_data.append((255, 255, 255))
                else:
                    test_image_data.append((0, 0, 0))
        img.putdata(test_image_data)
        img.save('verify_with_led.jpg')
        # print(time.time() - start_time)
        camera.close()

    def test(self, threshold=120, erode=1):
        """ testing whether the data input has been taken properly or not by assign 1 or 0 to each pixel random and
        then checking whether the computer can read the display accurately or not """
        # test_array contains the randomly selected led display data (rn its alternate pixels)
        test_array = np.array(np.zeros(64 * 128))
        test_array.shape = (64, 128)
        for i in range(20, 64, 2):
            for j in range(40, 128, 2):
                test_array[i][j] = int(np.random.randint(0,2,size = 1))

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

        filtered_list = self.filter(threshold, pixels2)
        img = Image.new('RGB', Image.open('test_crop.jpg').size)
        img.putdata(filtered_list)
        img.save('test_filter.jpg')
        img = cv2.imread('test_filter.jpg')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, None, iterations=erode)
        # result_array contains the data of the pixels that are being read
        result_array = np.array(np.zeros(64 * 128))
        result_array.shape = (64, 128)

        for i in range(0, 64, 2):
            for j in range(0, 128, 2):
                if gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] == 0:
                    result_array[i][j] = 0
                else:
                    result_array[i][j] = 1

        # checking whether result_array and test_array are in fact same or not
        errors_list = []
        errors = 0
        for i in range(0, 64, 2):
            for j in range(0, 128, 2):
                if (abs(result_array[i][j] - test_array[i][j])) > 0.1:
                    errors += 1
                    errors_list.append((i, j))

        print('errors', errors)

        img = Image.new('RGB', Image.open('test_filter.jpg').size)
        test_image_array = np.zeros(gray.shape)

        for i in range(64):
            for j in range(128):
                test_image_array[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] = 1
        test_image_data = []
        for i in range(test_image_array.shape[0]):
            for j in range(test_image_array.shape[1]):
                if test_image_array[i][j] == 1:
                    test_image_data.append((255, 0, 0))
                else:
                    test_image_data.append((gray[i][j], gray[i][j], gray[i][j]))
        img.putdata(test_image_data)
        img.save('verify2.jpg') # useful for checking

        camera.close()
    # ignore next two methods for now
    def test_1_with_threshold_array(self, num, threshold = None, erode = 2):
        """ randomly selected as many points as the parameter num to be bright pixels and tests whether they can
            be identified or not """
        if threshold == None:
            threshold = self.threshold_test_1
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

        img = Image.new('RGB', Image.open('test_crop.jpg').size)
        img.putdata(pixels)
        img.save('blue.jpg')
        img = cv2.imread('blue.jpg')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result_list = []
        for i in range(64):
            for j in range(128):
                if gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] > threshold[i][j]:
                    result_list.append((i, j))

        img = Image.new('RGB', Image.open('test_filter.jpg').size)
        test_image_array = np.zeros(gray.shape)

        for i in range(64):
            for j in range(128):
                test_image_array[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] = 1

        test_image_data = []
        for i in range(test_image_array.shape[0]):
            for j in range(test_image_array.shape[1]):
                if test_image_array[i][j] == 1:
                    test_image_data.append((255, 0, 0))
                else:
                    test_image_data.append((gray[i][j], gray[i][j], gray[i][j]))
        img.putdata(test_image_data)

        img.save('verify2.jpg')

        print('test_value', test_list, 'result_value', result_list)
        camera.close()
        return (test_list == result_list)

    def test_1_update_threshold(self, num = 1, threshold = None):
        """ randomly selected as many points as the parameter num to be bright pixels and tests whether they can
                be identified or not """
        if threshold == None:
            threshold = self.threshold_test_1
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

        img = Image.new('RGB', Image.open('test_crop.jpg').size)
        img.putdata(pixels)
        img.save('blue.jpg')
        img = cv2.imread('blue.jpg')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
        # these 3 lines are not really essential coz necessary work has been done already

        # gray = cv2.erode(gray, None, iterations=erode)
        result_list = []
        for i in range(64):
            for j in range(128):
                if (gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] > threshold[i][j]):
                    result_list.append((i, j))

        #img = Image.new('RGB', Image.open('test_filter.jpg').size)
        #test_image_array = np.zeros(gray.shape)
        #for i in range(64):
        #    for j in range(128):
         #       test_image_array[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] = 1
        #test_image_data = []
        #for i in range(test_image_array.shape[0]):
        #    for j in range(test_image_array.shape[1]):
        #        if test_image_array[i][j] == 1:
        #            test_image_data.append((255, 0, 0))
        #        else:
        #            test_image_data.append((gray[i][j], gray[i][j], gray[i][j]))
        #img.putdata(test_image_data)

       # img.save('verify2.jpg')
        camera.close()
        if len(result_list) == 0:
            self.threshold_test_1[test_list[0][0]][test_list[0][1]] -= 1

        return test_list==result_list

    def test_1(self, num, threshold=125, erode=2):
            """ randomly selected as many points as the parameter num to be bright pixels and tests whether they can
            be identified or not """
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

            filtered_list = self.filter(threshold, pixels2)
            img = Image.new('RGB', Image.open('test_crop.jpg').size)
            img.putdata(filtered_list)
            img.save('test_filter.jpg')
            img = cv2.imread('test_filter.jpg')

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.erode(gray, None, iterations=erode)
            result_list = []
            for i in range(64):
                for j in range(128):
                    if not (gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] == 0):
                        result_list.append((i, j))

            img = Image.new('RGB', Image.open('test_filter.jpg').size)
            test_image_array = np.zeros(gray.shape)
            for i in range(64):
                for j in range(128):
                    test_image_array[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] = 1
            test_image_data = []
            for i in range(test_image_array.shape[0]):
                for j in range(test_image_array.shape[1]):
                    if test_image_array[i][j] == 1:
                        test_image_data.append((255, 0, 0))
                    else:
                        test_image_data.append((gray[i][j], gray[i][j], gray[i][j]))
            img.putdata(test_image_data)

            if not test_list == result_list:
                img.save('verify2.jpg')

            print('test_value', test_list,'result_value', result_list)
            camera.close()
            return(test_list == result_list)

    def test_1_using_2_filters(self, num, threshold1=125, erode1=2, threshold2=125, erode2=2):
            """ randomly selected as many points as the parameter num to be bright pixels and tests whether they can
            be identified or not
            threshold1 , erode1 are values for the right side while threshold2 and erode2 are values for the left side
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
            camera.capture('test.jpg')
            camera.stop_preview()
            image = 'test.jpg'
            self.crop(image, (1020, 620, 1800, 1050), 'test_crop.jpg')
            image = 'test_crop.jpg'
            img = Image.open(image)
            pixels = list(img.getdata())
            pixels2 = []
            for pixel in pixels:
                pixels2.append((pixel[2], pixel[2], pixel[2]))  # taking only the blue pixels data

            result_list = []

            filtered_list = self.filter(threshold1, pixels2)
            img = Image.new('RGB', Image.open('test_crop.jpg').size)
            img.putdata(filtered_list)
            img.save('test_filter.jpg')
            img = cv2.imread('test_filter.jpg')

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.erode(gray, None, iterations=erode1)

            for i in range(64):
                for j in range(71):
                    if not (gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] == 0):
                        result_list.append((i, j))

            img = Image.new('RGB', Image.open('test_filter.jpg').size)
            test_image_array = np.zeros(gray.shape)
            for i in range(64):
                for j in range(128):
                    test_image_array[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] = 1
            test_image_data = []
            for i in range(test_image_array.shape[0]):
                for j in range(test_image_array.shape[1]):
                    if test_image_array[i][j] == 1:
                        test_image_data.append((255, 0, 0))
                    else:
                        test_image_data.append((gray[i][j], gray[i][j], gray[i][j]))
            img.putdata(test_image_data)

            if not test_list == result_list:
                img.save('verify2_1.jpg')

            filtered_list = self.filter(threshold2, pixels2)
            img = Image.new('RGB', Image.open('test_crop.jpg').size)
            img.putdata(filtered_list)
            img.save('test_filter.jpg')
            img = cv2.imread('test_filter.jpg')

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.erode(gray, None, iterations=erode2)

            for i in range(64):
                for j in range(71, 128, 1):
                    if not (gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] == 0):
                        result_list.append((i, j))

            img = Image.new('RGB', Image.open('test_filter.jpg').size)
            test_image_array = np.zeros(gray.shape)
            for i in range(64):
                for j in range(128):
                    test_image_array[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] = 1
            test_image_data = []
            for i in range(test_image_array.shape[0]):
                for j in range(test_image_array.shape[1]):
                    if test_image_array[i][j] == 1:
                        test_image_data.append((255, 0, 0))
                    else:
                        test_image_data.append((gray[i][j], gray[i][j], gray[i][j]))
            img.putdata(test_image_data)

            if not test_list == result_list:
                img.save('verify2_2.jpg')

            print('test_value', test_list,'result_value', result_list)
            camera.close()
            return (test_list == result_list)

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
        """ tries to read the input (from the led) and the associated weights (which is the display on the screen)"""
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


            filtered_list = self.filter(120, blue)
            img = Image.new('RGB', Image.open(image).size)
            img.putdata(filtered_list)
            img.save('input_filter.jpg')
            img = cv2.imread('input_filter.jpg')

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            # image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
            # these 3 lines are not really essential coz necessary work has been done already

            gray = cv2.erode(gray, None, iterations=1)
            # result_array contains the data of the pixels that are being read

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

    def test_with_led(self):
        """does the test method but with the led on instead"""
        led = LED(21)
        led.on()
        self.test(threshold=110, erode=0)

    def test_with_led_full_array(self, threshold=110, erode=0):
        """does the test method but with the led on and alternate pixels bright instead"""
        led = LED(21)
        led.on()
        # test_array contains the randomly selected led display data (rn its alternate pixels)
        test_array = np.array(np.zeros(64 * 128))
        test_array.shape = (64, 128)
        for i in range(0, 64, 2):
            for j in range(0, 128, 2):
                test_array[i][j] = 1

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

        filtered_list = self.filter(threshold, pixels2)
        img = Image.new('RGB', Image.open('test_crop.jpg').size)
        img.putdata(filtered_list)
        img.save('test_filter.jpg')
        img = cv2.imread('test_filter.jpg')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, None, iterations=erode)

        result_array = np.array(np.zeros((64, 128), dtype=int))

        for i in range(0, 64, 2):
            for j in range(0, 128, 2):
                if gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] == 0:
                    result_array[i][j] = 0
                else:
                    result_array[i][j] = 1
        # checking whether result_array and test_array are in fact same or not
        errors_list = []
        errors = 0
        for i in range(0, 64, 2):
            for j in range(0, 128, 2):
                if (abs(result_array[i][j] - test_array[i][j])) > 0.1:
                    errors += 1
                    errors_list.append((i, j))

        img = Image.new('RGB', Image.open('test_filter.jpg').size)
        test_image_array = np.zeros(gray.shape)
        for i in range(64):
            for j in range(128):
                test_image_array[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] = 1
        test_image_data = []
        for i in range(test_image_array.shape[0]):
            for j in range(test_image_array.shape[1]):
                if test_image_array[i][j] == 1:
                    test_image_data.append((255, 0, 0))
                else:
                    test_image_data.append((gray[i][j], gray[i][j], gray[i][j]))
        img.putdata(test_image_data)
        img.save('verify2.jpg')

        print('errors', errors)

        camera.close()

    def test_with_led_full_array_with_2_filters(self, threshold1=110, erode1=0, threshold2=110, erode2=0):
        """does the test method but with the led on and alternate pixels bright instead"""
        led = LED(21)
        led.on()
        # test_array contains the led display data (rn its alternate pixels)
        test_array = np.array(np.zeros((64, 128), dtype = int))
        for i in range(0, 64, 2):
            for j in range(0, 128, 2):
                test_array[i][j] = 1

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

        result_array = np.array(np.zeros((64, 128), dtype = int))

        filtered_list = self.filter(threshold1, pixels2)
        img = Image.new('RGB', Image.open('test_crop.jpg').size)
        img.putdata(filtered_list)
        img.save('test_filter.jpg')
        img = cv2.imread('test_filter.jpg')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, None, iterations=erode1)
        # result_array contains the data of the pixels that are being read

        for i in range(0, 64, 2):
            for j in range(0, 70, 2):
                if gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] == 0:
                    result_array[i][j] = 0
                else:
                    result_array[i][j] = 1

        img = Image.new('RGB', Image.open('test_filter.jpg').size)
        test_image_array = np.zeros(gray.shape)
        for i in range(64):
            for j in range(128):
                test_image_array[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] = 1
        test_image_data = []
        for i in range(test_image_array.shape[0]):
            for j in range(test_image_array.shape[1]):
                if test_image_array[i][j] == 1:
                    test_image_data.append((255, 0, 0))
                else:
                    test_image_data.append((gray[i][j], gray[i][j], gray[i][j]))
        img.putdata(test_image_data)
        img.save('verify2_1.jpg')

        filtered_list = self.filter(threshold2, pixels2)
        img = Image.new('RGB', Image.open('test_crop.jpg').size)
        img.putdata(filtered_list)
        img.save('test_filter.jpg')
        img = cv2.imread('test_filter.jpg')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, None, iterations=erode2)
        for i in range(0, 64, 2):
            for j in range(70, 128, 2):
                if gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] == 0:
                    result_array[i][j] = 0
                else:
                    result_array[i][j] = 1

        # checking whether result_array and test_array are in fact same or not
        errors_list = []
        errors = 0
        for i in range(0, 64, 2):
            for j in range(0, 128, 2):
                if (abs(result_array[i][j] - test_array[i][j])) > 0.1:
                    errors += 1
                    errors_list.append(((i, j), result_array[i][j]))

        img = Image.new('RGB', Image.open('test_filter.jpg').size)
        test_image_array = np.zeros(gray.shape)
        for i in range(64):
            for j in range(128):
                test_image_array[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] = 1
        test_image_data = []
        for i in range(test_image_array.shape[0]):
            for j in range(test_image_array.shape[1]):
                if test_image_array[i][j] == 1:
                    test_image_data.append((255, 0, 0))
                else:
                    test_image_data.append((gray[i][j], gray[i][j], gray[i][j]))
        img.putdata(test_image_data)
        img.save('verify2_2.jpg')

        print('errors', errors, errors_list)
# done
        camera.close()

    def test_1_with_led(self, num, threshold=110, erode=1):
        """does the test_1 method but with led on instead"""
        led = LED(21)
        led.on()
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
            pixels2.append((pixel[2], pixel[2], pixel[2]))

        filtered_list = self.filter(threshold, pixels2)
        img = Image.new('RGB', Image.open('test_crop.jpg').size)
        img.putdata(filtered_list)
        img.save('test_filter.jpg')
        img = cv2.imread('test_filter.jpg')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, None, iterations=erode)
        result_list = []
        for i in range(64):
            for j in range(128):
                if not (gray[int(self.all_points_array_with_led[i][j][0])][int(self.all_points_array_with_led[i][j][1])] == 0):
                    result_list.append((i, j))

        print('test_value', test_list, 'result_value', result_list)
        camera.close()
    # ignore next method
    def test_1_with_led_threshold_array(self, num, threshold = None):

            """ randomly selected as many points as the parameter num to be bright pixels and tests whether they can
                be identified or not """
            if threshold == None:
                threshold = self.threshold_test_1_with_led

            led = LED(21)
            led.on()
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

            img = Image.new('RGB', Image.open('test_crop.jpg').size)
            img.putdata(pixels)
            img.save('blue.jpg')
            img = cv2.imread('blue.jpg')

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
            # these 3 lines are not really essential coz necessary work has been done already

            # gray = cv2.erode(gray, None, iterations=erode)
            result_list = []
            for i in range(64):
                for j in range(128):
                    if (gray[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] > threshold[i][j]):
                        result_list.append((i, j))

            img = Image.new('RGB', Image.open('test_filter.jpg').size)
            test_image_array = np.zeros(gray.shape)
            for i in range(64):
                for j in range(128):
                    test_image_array[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] = 1
            test_image_data = []
            for i in range(test_image_array.shape[0]):
                for j in range(test_image_array.shape[1]):
                    if test_image_array[i][j] == 1:
                        test_image_data.append((255, 0, 0))
                    else:
                        test_image_data.append((gray[i][j], gray[i][j], gray[i][j]))
            img.putdata(test_image_data)

            img.save('verify2.jpg')

            print('test_value', test_list, 'result_value', result_list)
            camera.close()
            return (test_list == result_list)

    def test_1_with_led_part(self, num):
        """test_1 but with led = on"""
        led = LED(21)
        led.on()
        self.test_1(1, threshold=110, erode=1)

if __name__ == '__main__':
    disp = Disp()
    disp.start()

    #disp.start_with_led()

   # for i in range(20):
    #    disp.test_1(1,125,1)
    #    disp.test_with_led()
    start = time.time()
    for i in range(20):
        disp.test_with_led_full_array_with_2_filters(125, 0, 150, 0)
    print('time')
    print(time.time()- start)
    #correct = 0
    #for i in range(500):
  #      disp.test_1(1)
    #    print('test without led')
     #   if (disp.test_1_update_threshold()):
      #      correct += 1
        #print('test with led')
        #disp.test_1_with_led_part(1)
       # if (i+1)%50 == 0:
        #    print('correct' + str(correct))
         #   correct = 0
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
  #  weights = np.zeros(10*64*128)
  #  weights.shape = (10, 64, 128)
  #  input = []
#    for i in range(10):
  #      input.append(int(np.random.randint(0, 2, size=1)))
 #       for j in range(0, 64, 4):
   #         for k in range(0, 128, 2):
    #            weights[i][j][k] = 1 # int(np.random.randint(0, 2, size=1))

    #print('start')
    #start_time = time.time()
    #disp.read_input_stream(input, weights)
    #print('end_time = ' + str(time.time() - start_time))
