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
    # read with blue only

    def __init__(self):
        """ initialising the stuff
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
        self.min_array_without_led = np.array([])
        self.min_array_with_led = np.array([])
        self.max_array_without_led = np.array([])
        self.max_array_with_led = np.array([])

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
            if tup[0] + tup[1] + tup[2] > 3*cutoff_value:
                # if the pixel is above a certain RGB value then its a bright pixel else dark
                tup1 = tup
                # tup1 = (255, 255, 255)
            else:
                tup1 = (0, 0, 0)
            output_list.append(tup1)
        return output_list

    def start(self):
        """ In order to initialise the self.all_points_array  -  also going to be using the array
        at the end in order to define the threshold array """

        led= LED(21)
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

        # show the output image

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
        img.save('verify.jpg')

        self.min_array_without_led = np.zeros((self.height, self.width))
        self.min_array_with_led = np.zeros((self.height, self.width))
        self.max_array_without_led = np.zeros((self.height, self.width))
        self.max_array_with_led = np.zeros((self.height, self.width))

        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        # drawing a grid that is separated by 4 units in the vertical and 2 units in the horizontal
        for i in range(self.height):
            for j in range(self.width):
                draw.point([(j, i)], fill=255)

        disp.image(image_1)
        disp.display()

        led.off()
        img = 'threshold1_disp.jpg'
        image = 'threshold1_crop.jpg'
        camera.start_preview()
        camera.capture(img)
        camera.stop_preview()
        self.crop(img, (1020, 620, 1800, 1050), image)
        img1 = Image.open(image)
        pixels = list(img1.getdata())
        pixels2 = []
        for pixel in pixels:
            pixels2.append((pixel[2], pixel[2], pixel[2]))
        img2 = Image.new('RGB', Image.open(image).size)
        img2.putdata(pixels2)
        img2.save('threshold1_filter.jpg')
        image = 'threshold1_filter.jpg'
        im = cv2.imread(image)
        for i in range(self.height):
            for j in range(self.width):
                self.max_array_without_led[i][j] = im[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])][2]

        led.on()
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        # drawing a grid that is separated by 4 units in the vertical and 2 units in the horizontal
        for i in range(self.height):
            for j in range(self.width):
                draw.point([(j, i)], fill=255)

        disp.image(image_1)
        disp.display()

        img = 'threshold2_disp.jpg'
        image = 'threshold2_crop.jpg'
        # camera.start_preview()
        camera.capture(image)
        # camera.stop_preview()
        self.crop(img, (1020, 620, 1800, 1050), image)
        img1 = Image.open(image)
        pixels = list(img1.getdata())
        pixels2 = []
        for pixel in pixels:
            pixels2.append((pixel[2], pixel[2], pixel[2]))
        img2 = Image.new('RGB', Image.open(image).size)
        img2.putdata(pixels2)
        img2.save('threshold2_filter.jpg')
        image = 'threshold2_filter.jpg'
        im = cv2.imread(image)
        for i in range(self.height):
            for j in range(self.width):
                self.max_array_with_led[i][j] = im[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])][2]

        led.on()
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        disp.image(image_1)
        disp.display()
        img = 'threshold3_disp.jpg'
        image = 'threshold3_crop.jpg'
        camera.start_preview()
        camera.capture(image)
        camera.stop_preview()
        self.crop(img, (1020, 620, 1800, 1050), image)
        img1 = Image.open(image)
        pixels = list(img1.getdata())
        pixels2 = []
        for pixel in pixels:
            pixels2.append((pixel[2], pixel[2], pixel[2]))
        img2 = Image.new('RGB', Image.open(image).size)
        img2.putdata(pixels2)
        img2.save('threshold3_filter.jpg')
        image = 'threshold3_filter.jpg'
        im = cv2.imread(image)
        for i in range(self.height):
            for j in range(self.width):
                self.min_array_with_led[i][j] = im[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])][2]

        led.off()
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        disp.image(image_1)
        disp.display()
        img = 'threshold4_disp.jpg'
        image = 'threshold4_crop.jpg'
        camera.start_preview()
        camera.capture(image)
        camera.stop_preview()
        self.crop(img, (1020, 620, 1800, 1050), image)
        img1 = Image.open(image)
        pixels = list(img1.getdata())
        pixels2 = []
        for pixel in pixels:
            pixels2.append((pixel[2], pixel[2], pixel[2]))
        img2 = Image.new('RGB', Image.open(image).size)
        img2.putdata(pixels2)
        img2.save('threshold4_filter.jpg')
        image = 'threshold4_filter.jpg'
        im = cv2.imread(image)
        for i in range(self.height):
            for j in range(self.width):
                self.min_array_without_led[i][j] = im[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])][2]

        camera.close()

    def start_1(self):
        """ In order to initialise the self.all_points_array  -  also going to be using the array
        at the end in order to define the threshold array """

        self.all_points_array = np.zeros(64 * 128 * 2)
        self.all_points_array.shape = (64, 128, 2)

        led= LED(21)
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
        time.sleep(.5)
        camera.capture('full_multi_point_2.jpg')
        camera.stop_preview()
        # taking a picture of the display
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

        for i in range(0, 16):
            for j in range(0, 64):
                self.all_points_array[60 - 4 * i][126 - 2 * j][0] = new_pixel_array[i][j][0]
                self.all_points_array[60 - 4 * i][126 - 2 * j][1] = new_pixel_array[i][j][1]
        # finding the coords of the bright pixels
        for i in range(0, 61, 4):
            for j in range(1, 127, 2):
                    self.all_points_array[i][j][0] = int((self.all_points_array[i][j - 1][0] + self.all_points_array[i][j + 1][0]) / 2)
                    self.all_points_array[i][j][1] = int((self.all_points_array[i][j - 1][1] + self.all_points_array[i][j + 1][1]) / 2)

        print('A')
        # Draw a black filled box to clear the image.
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        # drawing a grid that is separated by 4 units in the vertical and 2 units in the horizontal
        for i in range(2, self.height, 4):
            for j in range(0, self.width, 2):
                draw.point([(j, i)], fill=255)  # x,y

        disp.image(image_1)
        disp.display()

        camera.start_preview()
        camera.led = False
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

        filtered_list = self.filter(140, pixels2)  # cropped and filtered that image
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

        for i in range(0, 16):
            for j in range(0, 64):
                self.all_points_array[62 - 4 * i][126 - 2 * j][0] = new_pixel_array[i][j][0]
                self.all_points_array[62 - 4 * i][126 - 2 * j][1] = new_pixel_array[i][j][1]
        # finding the coords of the bright pixels
        for i in range(2, 63, 4):
            for j in range(1, 127, 2):
                    self.all_points_array[i][j][0] = int(
                        (self.all_points_array[i][j - 1][0] + self.all_points_array[i][j + 1][0]) / 2)
                    self.all_points_array[i][j][1] = int(
                        (self.all_points_array[i][j - 1][1] + self.all_points_array[i][j + 1][1]) / 2)

        # interpolation

        for i in range(1, 62, 2):
            for j in range(127):
                self.all_points_array[i][j][0] = int(
                        (self.all_points_array[i - 1][j][0] + self.all_points_array[i + 1][j][0]) / 2)
                self.all_points_array[i][j][1] = int(
                        ( self.all_points_array[i - 1][j][1] + self.all_points_array[i + 1][j][1]) / 2)

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
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pixel_point.append((int(cY), int(cX)))

        pixel_point_sort = sorted(pixel_point, key=lambda z: z[1])

        for j in range(64):
            self.all_points_array[63][126 - 2 * j][0] = pixel_point_sort[j][0]
            self.all_points_array[63][126 - 2 * j][1] = pixel_point_sort[j][1]

        i = 63
        for j in range(1,126,2):
            self.all_points_array[i][j][1] = (self.all_points_array[i][j-1][1] + self.all_points_array[i][j + 1][1])/ 2
            self.all_points_array[i][j][0] = (self.all_points_array[i][j-1][0] + self.all_points_array[i][j+1][0])/2

        # now drawing the last column
        j = 127

        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        for i in range(0, self.width, 3):
            draw.point([(j, i)], fill=255)  # x,y

        disp.image(image_1)
        disp.display()

        # Now get the pixel data

        camera.start_preview()
        camera.led = False
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

        # show the output image

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
        img.save('verify.jpg')

        self.min_array_without_led = np.zeros((self.height, self.width))
        self.min_array_with_led = np.zeros((self.height, self.width))
        self.max_array_without_led = np.zeros((self.height, self.width))
        self.max_array_with_led = np.zeros((self.height, self.width))

        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        # drawing a grid that is separated by 4 units in the vertical and 2 units in the horizontal
        for i in range(self.height):
            for j in range(self.width):
                draw.point([(j, i)], fill=255)

        disp.image(image_1)
        disp.display()

        led.off()
        img = 'threshold1_disp.jpg'
        image = 'threshold1_crop.jpg'
        camera.capture(img)
        self.crop(img, (1020, 620, 1800, 1050), image)
        img1 = Image.open(image)
        pixels = list(img1.getdata())
        pixels2 = []
        for pixel in pixels:
            pixels2.append((pixel[2], pixel[2], pixel[2]))
        img2 = Image.new('RGB', Image.open(image).size)
        img2.putdata(pixels2)
        img2.save('threshold1_filter.jpg')
        image = 'threshold1_filter.jpg'
        im = cv2.imread(image)
        for i in range(self.height):
            for j in range(self.width):
                self.max_array_without_led[i][j] = im[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])][2]

        led.on()
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        # drawing a grid that is separated by 4 units in the vertical and 2 units in the horizontal
        for i in range(self.height):
            for j in range(self.width):
                draw.point([(j, i)], fill=255)

        disp.image(image_1)
        disp.display()

        img = 'threshold2_disp.jpg'
        image = 'threshold2_crop.jpg'
        camera.start_preview()
        time.sleep(.2)
        camera.capture(image)
        camera.stop_preview()
        self.crop(img, (1020, 620, 1800, 1050), image)
        img1 = Image.open(image)
        pixels = list(img1.getdata())
        pixels2 = []
        for pixel in pixels:
            pixels2.append((pixel[2], pixel[2], pixel[2]))
        img2 = Image.new('RGB', Image.open(image).size)
        img2.putdata(pixels2)
        img2.save('threshold2_filter.jpg')
        image = 'threshold2_filter.jpg'
        im = cv2.imread(image)
        for i in range(self.height):
            for j in range(self.width):
                if j in [63, 127] or i in [63]:
                    self.max_array_with_led[i][j] = 150
                else:
                    self.max_array_with_led[i][j] = im[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])][2]

        led.on()
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        disp.image(image_1)
        disp.display()
        img = 'threshold3_disp.jpg'
        image = 'threshold3_crop.jpg'
        camera.capture(image)
        self.crop(img, (1020, 620, 1800, 1050), image)
        img1 = Image.open(image)
        pixels = list(img1.getdata())
        pixels2 = []
        for pixel in pixels:
            pixels2.append((pixel[2], pixel[2], pixel[2]))
        img2 = Image.new('RGB', Image.open(image).size)
        img2.putdata(pixels2)
        img2.save('threshold3_filter.jpg')
        image = 'threshold3_filter.jpg'
        im = cv2.imread(image)
        for i in range(self.height):
            for j in range(self.width):
                self.min_array_with_led[i][j] = im[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])][2]

        led.off()
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        disp.image(image_1)
        disp.display()
        img = 'threshold4_disp.jpg'
        image = 'threshold4_crop.jpg'
        camera.capture(image)
        self.crop(img, (1020, 620, 1800, 1050), image)
        img1 = Image.open(image)
        pixels = list(img1.getdata())
        pixels2 = []
        for pixel in pixels:
            pixels2.append((pixel[2], pixel[2], pixel[2]))
        img2 = Image.new('RGB', Image.open(image).size)
        img2.putdata(pixels2)
        img2.save('threshold4_filter.jpg')
        image = 'threshold4_filter.jpg'
        im = cv2.imread(image)
        for i in range(self.height):
            for j in range(self.width):
                self.min_array_without_led[i][j] = im[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])][2]

        print(self.max_array_with_led.max(), self.max_array_with_led.min() , self.max_array_with_led.mean(), 'max with led' )
        print(self.max_array_without_led.max(), self.max_array_without_led.min(), self.max_array_without_led.mean(),
              'max without led')
        print(self.min_array_with_led.max(), self.min_array_with_led.min(), self.min_array_with_led.mean(),
              'min with led')
        print(self.min_array_without_led.max(), self.min_array_without_led.min(), self.min_array_without_led.mean(),
              'min without led')

        low = []
        for i in range(self.max_array_with_led.shape[0]):
            for j in range(self.max_array_with_led.shape[1]):
                if self.max_array_with_led[i][j] < 120 and j not in [63, 127] and i not in [63]:
                    low.append((i, j))
        print(low)
        camera.close()

    def read_led(self, input, num):
        led = LED(21)
        led.off()
        # if input == 1:
          #  led.on()
        #else:
         #   led.off()
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

        test_array = np.zeros((self.height, self.width), dtype=int)
        list_test = []
        repeat = True
        for k in range(num):
            while repeat:
                i, j = np.random.randint(0, self.height-1, 1)[0], np.random.randint(0, self.width-1, 1)[0]
                if i not in [62, 63] and j not in [127, 63, 64]:
                    break
            list_test.append((i, j))
            test_array[i][j] = 1

        # Draw a black filled box to clear the image.
        draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        # drawing a grid that is separated by 4 units in the vertical and 2 units in the horizontal
        for i in range(0, self.height):
            for j in range(0, self.width):
                if test_array[i][j] > 0:
                    draw.point([(j, i)], fill=255)  # x,y

        disp.image(image_1)
        disp.display()
        # made the display needed to find the points

        # Now get the pixel data

        camera = picamera.PiCamera()
        camera.resolution = (2592, 1944)
        camera.start_preview()
        camera.led = False
        camera.capture('full_multi_point_2.jpg')
        camera.stop_preview()   # taking a picture of the display
        image = 'full_multi_point_2.jpg'
        self.crop(image, (1020, 620, 1800, 1050), 'full_multi_crop_2.jpg')
        image = 'full_multi_crop_2.jpg'
        img = cv2.imread(image) # this gives BGR
        blue = img[:, :, 0]
        red = img[:, :, 2]
        if red.mean() > 150:
            led_input = 1
        else: led_input = 0

        blue_data = np.zeros((self.height, self.width), dtype=float)
        for i in range(self.height - 2):
            for j in range(self.width - 1):
                blue_data[i][j] = blue[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])]

       # if led_input == 1:
           # signal = blue_data / self.max_array_with_led
            # signal = (blue_data - self.min_array_with_led)/(self.max_array_with_led - self.min_array_with_led)
       # else:
        signal = blue_data / self.max_array_without_led
        #   signal = (blue_data - self.min_array_without_led)/(self.max_array_without_led - self.min_array_without_led)
        list_points = []
        result_array = np.zeros((self.height, self.width), dtype=int)
        for k in range(self.height - 2):
            for j in range(self.width - 1):
                if j not in [63, 64]:
                    # if signal[k][j] > 1:
                    if blue_data[k][j] > 190:
                        list_points.append(((k, j), blue_data[k][j]))
                        result_array[k][j] = 1

        error_list = []
        for i in range(self.height):
            for j in range(self.width):
                if not result_array[i][j] == test_array[i][j]:
                    error_list.append((i, j))

        print('error list', error_list)
        for a in error_list:
            print(a, a in list_test)
            print(signal[a[0]][a[1]])
            print(blue_data[a[0]][a[1]])
        print(str(len(error_list)) + str(input == led_input) + str(input))
        print('')
        camera.close()

        img = Image.new('RGB', Image.open('test_filter.jpg').size)
        test_image_array = np.zeros(blue.shape)
        for i in range(64):
            for j in range(128):
                test_image_array[int(self.all_points_array[i][j][0])][int(self.all_points_array[i][j][1])] = 1
        test_image_data = []
        for i in range(test_image_array.shape[0]):
            for j in range(test_image_array.shape[1]):
                if test_image_array[i][j] == 1:
                    test_image_data.append((255, 0, 0))
                else:
                    test_image_data.append((blue[i][j], blue[i][j], blue[i][j]))
        img.putdata(test_image_data)
        img.save('verify_new.jpg')

        return len(error_list)


if __name__ == '__main__':
    disp = Disp()
    disp.start_1()
    start = time.time()
    for i in range(20):
        if disp.read_led(np.random.randint(0, 2, size=1), 50) > 0:
            break

    print(time.time() - start)
