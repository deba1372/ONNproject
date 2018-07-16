from PIL import Image
import matplotlib.pyplot as py
import numpy as np

# Just doing this to crop the image taken to a set of dimensions

def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.show()


if __name__ == '__main__':
    image = 'last_col_disp.jpg'
    crop(image, (1020, 620, 1800, 1050), 'last_col_crop.jpg')


