from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image
import time
import random

# in order to read the pixels in the last row

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the image file")
# args = vars(ap.parse_args())
start_time = time.time()

image = cv2.imread('last_row_filter.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (11, 11), 0)
# image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
# these 3 lines are not really essential coz necessary work has been done already

gray = cv2.erode(gray, None, iterations=1)

#image = cv2.dilate(image, None, iterations=4)

new_image = Image.fromarray(gray)

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
    (x, y, w, h) = cv2.boundingRect(c)
    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
    pixel_point.append((int(cY), int(cX)))
    cv2.circle(image, (int(cX), int(cY)), int(radius),(0, 0, 255), 2)
    # cv2.putText(image, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image

pixel_point_sort = sorted(pixel_point, key=lambda x: x[1])

print(pixel_point_sort)

print(time.time() - start_time)