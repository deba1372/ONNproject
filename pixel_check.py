from PIL import Image
import matplotlib.pyplot as py
import numpy as np

# this program is just for testing the max and min of the image as well as to test
# whether the filter method is working

def crop(image_path, coords, saved_location):
    """
    crops image to given coordinates and saves it
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    # cropped_image.show()

def filter(image_name, cutoff, new_name):
    """ makes an image with pixels that are above the cutoff(value) only"""
    image_obj = py.imread(image_name)
    image_list = image_obj.tolist()
    new_image_list = []
    for list1 in image_list:
        new_list1 = []
        for list2 in list1:
                new_list2 = []
                pixel_valid = False
                for pixel in list2:
                    if pixel >= cutoff:
                        pixel_valid = True
                    new_list2.append(pixel)
                if pixel_valid:
                    new_list1.append(new_list2)
                else:
                    new_list1.append([0,0,0])
        new_image_list.append(new_list1)
    new_image_obj = np.array(new_image_list)
    print(type(new_image_obj))
    im = Image.fromarray(new_image_obj, 'RGB')
    im.save(new_name)


if __name__ == '__main__':
    image = 'pixel.jpg'
    crop(image, (1100, 620, 1900, 1050), 'cropped.jpg')
    img = py.imread('cropped.jpg')
    print('Shape for Cropped =' , img.shape)
    print('max for Cropped = ', img.max())
    print('min for Cropped = ' , img.min())
    print( 'mean for Cropped = ' , img.mean())


    img_new = img.sum(axis=2)
    img_new2 = []
    for i in range(430):
        for j in range(800):
            img_new2.append((int(img_new[i][j]/3),int(img_new[i][j]/3),int(img_new[i][j]/3)))   # averaging all axxes

    img = Image.new('RGB', Image.open('pixel.jpg').size)
    img.putdata(img_new2)
    img.save('filtered.jpg')
