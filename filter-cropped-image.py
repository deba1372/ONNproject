from PIL import Image
import matplotlib.pyplot as py
import numpy as np

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



if __name__ == '__main__':
    # in order to take an image and then filter it - every pixel whose average brightness is above a certain
    # threshold is white and less than that is black
    image = 'last_col_crop.jpg'
    img = Image.open(image)
    pixels = list(img.getdata())
    pixels2 = []
    for pixel in pixels:
        total = 0
        for x in pixel:
            total += x
        total = int(total/3)
        pixels2.append((total,total,total))

    filtered_list = filter(140, pixels2)
    img = Image.new('RGB', Image.open(image).size)
    img.putdata(filtered_list)
    img.show()
    img.save('last_col_filter.jpg')
