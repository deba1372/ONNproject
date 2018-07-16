from PIL import Image
import matplotlib.pyplot as py
import numpy as np


##want to find the cornermost pixels

def mean_bright_pt(pixels_array, start_height, end_height, start_width, end_width):
    '''returns the coordinates of a bright_pt present in the region bound by the coords provided'''
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
    corner_height = int(total_height / num_pts)
    corner_width = int(total_width / num_pts)
    return ((corner_height, corner_width))


if __name__ == '__main__':
    # takes an image of the display with only the corner points lit up
    # and then uses that to find the centers of the corner pixels

    image = 'filtered_only_corner.jpg'
    img = Image.open(image)
    pixels = list(img.getdata())
    pixels_array = py.imread(image)
    height, width, depth = pixels_array.shape

    corner_1_height, corner_1_width = mean_bright_pt(pixels_array, 0, int(height / 2), 0, int(width / 2))
    corner_2_height, corner_2_width = mean_bright_pt(pixels_array, 0, int(height / 2), int(width / 2) + 1, width)
    corner_3_height, corner_3_width = mean_bright_pt(pixels_array, int(height / 2) + 1, height, 0, int(width / 2))
    corner_4_height, corner_4_width = mean_bright_pt(pixels_array, int(height / 2) + 1, height, int(width / 2) + 1,
                                                     width)

    print(corner_1_height, corner_1_width)
    print(corner_2_height, corner_2_width)
    print(corner_3_height, corner_3_width)
    print(corner_4_height, corner_4_width)
    corner_pt_list = [(corner_1_height, corner_1_width),(corner_2_height, corner_2_width),(corner_3_height, corner_3_width),(corner_4_height, corner_4_width)]

    new_pixel_list = []
    for i in range(height):
        for j in range(width):
            if  (i,j) in corner_pt_list:
                print(True)
                new_pixel_list.append((255, 255, 255))
            else:
                new_pixel_list.append((0, 0, 0))
    img = Image.new('RGB', Image.open(image).size)
    img.putdata(new_pixel_list)
    img.save('corner_pts.jpg')
    #this image points out where the cornermost pixels are