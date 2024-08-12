from src.utility import process_image_set

if __name__ == '__main__':
    images_directory = '../../test_images'  # change to your image directory

    output_directory = '../../test_output'  # change to the directory you want to save the cropped images to

    x_max_crop = 14200  # max image size in the x

    y_max_crop = 14200  # max image size in the y

    process_image_set(images_directory, output_directory, x_max_crop, y_max_crop)
