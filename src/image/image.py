from statistics import pvariance

import cv2
import os
from typing import Optional
import numpy as np
import pyexiv2


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point(x={self.x}, y={self.y}"


class PrincipalPoint(Point):
    def __str__(self):
        return f"PrincipalPoint(x={self.x}, y={self.y})"


class Image:
    def __init__(self, file_path):
        self.filename = os.path.basename(file_path)
        self.name = os.path.splitext(self.filename)[0]
        self.original_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        self.exif_image = pyexiv2.Image(file_path)

        self.principal_point: Optional[PrincipalPoint] = None
        self.original_image_cropped = None
        self.fiducial_points = None

    def set_principal_point(self, x, y):
        self.principal_point = PrincipalPoint(x, y)

    def crop_original(self, x_max, y_max):
        if x_max > self.image.shape[1] or y_max > self.image.shape[0]:
            print('ERROR: crop dimensions are larger than original image')
        elif self.principal_point is None:
            print('ERROR: principal point is unknown!')

        else:
            # Calculate the top-left corner of the crop
            x_start = max(self.principal_point.x - x_max // 2, 0)
            y_start = max(self.principal_point.y - y_max // 2, 0)

            # Calculate the bottom-right corner of the crop, ensuring it doesn't exceed the image bounds
            x_end = min(self.principal_point.x + x_max // 2, self.original_image.shape[1])
            y_end = min(self.principal_point.y + y_max // 2, self.original_image.shape[0])

            # Crop the image using array slicing
            self.original_image_cropped = self.original_image[y_start:y_end, x_start:x_end]

    def crop_and_rotate(self, x_max, y_max):
        pass

    def save_original_image_cropped(self, save_file_path: str):
        cv2.imwrite(save_file_path, self.original_image_cropped,
                    [cv2.IMWRITE_TIFF_COMPRESSION, 1])

        image = pyexiv2.Image(save_file_path)
        self.exif_image.copy_to_another_image(image, exif=True, iptc=True, xmp=True, comment=False, icc=False,
                                              thumbnail=False)
        image.close()

    def display_original_image(self, scale_percent=4):
        self._display_image(self.original_image, scale_percent)

    def display_original_image_cropped(self, scale_percent=4):
        self._display_image(self.original_image_cropped, scale_percent)

    def save_fiducial_data_thumbnail(self, save_file_path):
        image = self.image.copy()

        # Scale the copied image to 5% of its original size
        scale_percent = 10  # percentage of the original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        image = self.draw_line(image, self.fiducial_points.get("top").x, self.fiducial_points.get("top").y,
                               self.fiducial_points.get("bottom").x, self.fiducial_points.get("bottom").y)

        image = self.draw_line(image, self.fiducial_points.get("left").x, self.fiducial_points.get("left").y,
                               self.fiducial_points.get("right").x, self.fiducial_points.get("right").y)

        # Resize the image
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        cv2.imwrite(save_file_path, image)

        # cv2.imshow('Scaled Image with Lines', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    @staticmethod
    def draw_line(image, x1, y1, x2, y2, color=(0, 0, 0), thickness=6):
        """
        Draws a line on an image.

        Parameters:
        - image: The image on which to draw the line.
        - x1, y1: Coordinates of the starting point of the line.
        - x2, y2: Coordinates of the ending point of the line.
        - color: Color of the line (BGR tuple, e.g., (255, 0, 0) for blue).
        - thickness: Thickness of the line in pixels.

        Returns:
        - The image with the line drawn on it.
        """
        # Draw the line on the image
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

        return image

    @staticmethod
    def _display_image(image, scale_percent):
        """
        Resizes and displays the image based on the scale percentage.
        Parameters:
        - image: The input image to be resized and displayed.
        - scale_percent: The percentage to scale the image (default is 4%).
        """
        try:
            # If the image is 16-bit, scale it down to 8-bit for visualization
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)  # Convert 16-bit to 8-bit by scaling

            original_image = image.copy()
            aspect_ratio = original_image.shape[1] / original_image.shape[0]

            # Calculate initial dimensions
            new_width = int(original_image.shape[1] * scale_percent / 100)
            new_height = int(original_image.shape[0] * scale_percent / 100)

            # Resize the image initially to the specified scale
            resized_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Convert to 3 channels if the image is grayscale
            if len(resized_image.shape) == 2:  # Grayscale image has 2 dimensions
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

            # Create a named window that can be resized
            window_name = f'Resized Image ({scale_percent}%)'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, resized_image)

            while True:
                key = cv2.waitKey(10)
                if key == 27:  # Escape key
                    break

                try:
                    # Check if the window is still open
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break

                    # Get the current size of the window
                    window_width, window_height = cv2.getWindowImageRect(window_name)[2:4]

                    # Determine the new dimensions for the image, keeping the aspect ratio
                    if window_width / window_height > aspect_ratio:
                        display_height = window_height
                        display_width = int(display_height * aspect_ratio)
                    else:
                        display_width = window_width
                        display_height = int(display_width / aspect_ratio)

                    # Resize the image to fit the new dimensions
                    resized_image = cv2.resize(original_image, (display_width, display_height),
                                               interpolation=cv2.INTER_AREA)

                    # Convert to 3 channels if the image is grayscale
                    if len(resized_image.shape) == 2:  # Grayscale image has 2 dimensions
                        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

                    # Create a black canvas of the size of the window
                    canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)

                    # Place the resized image in the center of the canvas
                    y_offset = (window_height - display_height) // 2
                    x_offset = (window_width - display_width) // 2
                    canvas[y_offset:y_offset + display_height, x_offset:x_offset + display_width] = resized_image

                    # Display the canvas
                    cv2.imshow(window_name, canvas)

                except cv2.error:
                    break

            cv2.destroyAllWindows()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    test_path = "../../test_images/5V-151.tif"
    image = Image(test_path)
    print('ok')

    image.display_original_image()

    image.set_principal_point(10000, 10000)

    print(image.principal_point)

    image.crop_original(1000, 1000)

    image.display_original_image_cropped()
