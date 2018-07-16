from PIL import Image
import matplotlib.pyplot as py
import numpy as np


##NOT A GOOD PROGRAM _ IGNORE

##want to find the cornermost pixels

def filter(cutoff_value , input_list):
    """takes in a list of tuples containing pixel data dn returns (255,255,255)
    for every pixel that is greater than cutoff and (0,0,0) for each pixel less than
    cutoff"""
    output_list = []
    for tup in input_list:
        if tup[0] > cutoff_value:     # if pixel value is greater than cutoff then it is made a bright pixel else dark
            tup1 = (255,255,255)
        else:
            tup1 = (0,0,0)
        output_list.append(tup1)
    return output_list


def mean_bright_pt(pixels_array, start_height, end_height, start_width, end_width):
    """ returns the coordinates of a bright_pt """
    # finds the average of all the bright points by giving a weight of 1 to any bright pixel and 0 otherwise
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

def get_filtered_array(cutoff):
    image = 'cropped-whole-array-ones.jpg'
    img = Image.open(image)
    pixels = list(img.getdata())
    pixels2 = []
    for pixel in pixels:
        total = 0
        for x in pixel:
            total += x
        total = int(total/3)
        pixels2.append((total, total, total))
    # takes the image and filters it based on input cutoff
    filtered_list = filter(cutoff, pixels2)



if __name__ == '__main__':
    image = 'filtered_only_corner.jpg'
    img = Image.open(image)
    pixels = list(img.getdata())
    pixels_array = py.imread(image)
    height, width, depth = pixels_array.shape
    get_filtered_array(140)
    corner_1_height, corner_1_width = mean_bright_pt(pixels_array, 0, int(height / 2), 0, int(width / 2))  # top left
    corner_2_height, corner_2_width = mean_bright_pt(pixels_array, 0, int(height / 2), int(width / 2) + 1, width)
    # top right
    corner_3_height, corner_3_width = mean_bright_pt(pixels_array, int(height / 2) + 1, height, 0, int(width / 2))
    # bottom left
    corner_4_height, corner_4_width = mean_bright_pt(pixels_array, int(height / 2) + 1, height, int(width / 2) + 1,
                                                     width)  # bottom right
    # found the four corners of the image
    row_1_y_fl = list(np.linspace(corner_1_height,corner_2_height,127))
    row_1_x_fl = list(np.linspace(corner_1_width,corner_2_width,127))
    # row is made by linear interpolation
    row_1_y = [int(y) for y in row_1_y_fl]
    row_1_x = [int(x) for x in row_1_x_fl]

    row_1_vert = []
    for i in range(127):
        row_1_vert.append((row_1_y[i], row_1_x[i]))

    row_63_y_fl = list(np.linspace(corner_3_height, corner_4_height, 127))
    row_63_x_fl = list(np.linspace(corner_3_width, corner_4_width, 127))
    # again linear interpolation
    row_63_y = [int(y) for y in row_63_y_fl]
    row_63_x = [int(x) for x in row_63_x_fl]

    row_63_vert = []
    for i in range(127):
        row_63_vert.append((row_63_y[i], row_63_x[i]))

    points_grid = []

    for i in range(63):
        points_grid.append([])

    for i in range(127):
        column_i_y_fl = list(np.linspace(row_1_vert[i][0],row_63_vert[i][0], 63))
        column_i_x_fl = list(np.linspace(row_1_vert[i][1], row_63_vert[i][1], 63))

        column_i_y = [int(y) for y in column_i_y_fl]
        column_i_x = [int(x) for x in column_i_x_fl]
    # formed a grid by linear interpolation
        for j in range(63):
            points_grid[j].append((column_i_y[j],column_i_x[j]))
    # using the image for test
    image_obj = py.imread('filtered-whole-array-ones.jpg')
    count = 0
    # count keeps track of the number of bright points in the image (ignore)
    a, b, c = image_obj.shape
    print(a, b)
    for i in range(a):
        for j in range(b):
            if image_obj[i][j][0] == 255:
                count += 1
    print(count)

    temp = np.ones(127)
    read_data = []
    for i in range(63):
        read_data.append(temp)
    # creates a 127 x 63 array of ones

    for i in range(63):
        for j in range(127):
            if image_obj[points_grid[i][j][0]][points_grid[i][j][1]][0] == 0:
                read_data[i][j] = 0
    # stores in the array read_data whether the pixel is bright or not

    count = 0
    for i in range(63):
        for j in range(127):
            if read_data[i][j] == 0:
                count += 1
    # counts the number of bright pixels

    print(count)
