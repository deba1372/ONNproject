from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image
import time
import random

# in order to identify the center points of the pixels of the display

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the image file")
# args = vars(ap.parse_args())
start_time = time.time()

image = cv2.imread('full_multi_filter_2.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (11, 11), 0)
# image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
# these 3 lines are not really essential coz necessary work has been done already

gray = cv2.erode(gray, None, iterations=1)

#image = cv2.dilate(image, None, iterations=4)

new_image = Image.fromarray(gray)
new_image.save('cv_proc_mult.png')

start_time_1 = time.time()

labels = measure.label(gray, neighbors=8, background=0)
mask = np.zeros(gray.shape, dtype="uint8")

print('check 1' ,  time.time() - start_time_1)

start_time_2 = time.time()
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
print('check 2' , time.time() - start_time_2)

pixel_point = []
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = contours.sort_contours(cnts)[0]
# loop over the contours
for (i, c) in enumerate(cnts):
    # draw the bright spot on the image
    (x, y, w, h) = cv2.boundingRect(c)
    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
    pixel_point.append((int(cY), int(cX)))
    cv2.circle(image, (int(cX), int(cY)), int(radius),(0, 0, 255), 2)
    # cv2.putText(image, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
new_image = Image.fromarray(image)
new_image.save('full_cv_proc_mult_2.jpg')

pixel_point_sort = sorted(pixel_point, key=lambda x: x[0])
new_pixel_point = []
total = len(pixel_point)
for i in range(16):
        new_pixel_point.append(sorted(pixel_point_sort[64*i:64*(i+1)], key = lambda x :x[1]))

new_pixel_array = np.array([y for x in new_pixel_point for y in x])
new_pixel_array.shape = (16, 64, 2)

all_points_array = np.array(np.zeros(61*127*2))
all_points_array.shape = (61, 127, 2)
for i in range(0, 16):
    for j in range(0, 64):
        all_points_array[60-4*i][126-2*j][0] = new_pixel_array[i][j][0]
        all_points_array[60-4*i][126-2*j][1] = new_pixel_array[i][j][1]

for i in range(0, 61, 4):
    for j in range(1, 127, 2):
        if j%2==1:
            all_points_array[i][j][0] = int((all_points_array[i][j-1][0] + all_points_array[i][j+1][0])/2)
            all_points_array[i][j][1] = int((all_points_array[i][j-1][1] + all_points_array[i][j+1][1])/2)

for i in range(61):
    for j in range(127):
        if i%4 == 1:
            all_points_array[i][j][0] = int((3*all_points_array[i-1][j][0] + all_points_array[i+3][j][0]) / 4)
            all_points_array[i][j][1] = int((3*all_points_array[i-1][j][1] + all_points_array[i+3][j][1]) / 4)
        elif i%4 == 2:
            all_points_array[i][j][0] = int((2 * all_points_array[i - 2][j][0] + 2*all_points_array[i + 2][j][0]) / 4)
            all_points_array[i][j][1] = int((2 * all_points_array[i - 2][j][1] + 2*all_points_array[i + 2][j][1]) / 4)
        elif i%4 == 3:
            all_points_array[i][j][0] = int(( all_points_array[i - 3][j][0] + 3*all_points_array[i + 1][j][0]) / 4)
            all_points_array[i][j][1] = int(( all_points_array[i - 3][j][1] + 3*all_points_array[i + 1][j][1]) / 4)

error = []
for i in range(61):
    for j in range(127):
        if i % 4 == 0 and j%2 == 0:
            if int(gray[int(all_points_array[i][j][0])][int(all_points_array[i][j][1])]) == 0:
                error.append((i,j))
        else:
            if int(gray[int(all_points_array[i][j][0])][int(all_points_array[i][j][1])]) > 0:
                error.append((i,j))

print(len(error))
print(time.time() - start_time)
