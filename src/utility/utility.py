import os
from functools import partial
from multiprocessing import Pool

import cv2
from tqdm import tqdm

from src.fiducial import FiducialTemplate
from src.image import Image, PrincipalPoint, Point


def process_single_image(image_path, fiducial_templates, x_crop, y_crop, output_directory, output_thumbnails):
    if '.tif' in image_path:
        image = Image(image_path)

        # try all fiducial templates on the image
        results = find_fiducials(image.image, fiducial_templates)

        # get the best scoring result
        best_matching_result = get_top_scores(results)
        print(f"\nBest matching result:\n{best_matching_result}\n")

        # get the points for the fiducial markers
        fiducial_points = get_fiducial_points(best_matching_result)
        image.fiducial_points = fiducial_points

        # get the principal point
        image.principal_point = compute_principal_point(fiducial_points)
        print(f"{image.principal_point}\n")

        # crop the image with the principal point as the center
        image.crop_original(x_crop, y_crop)

        save_file = os.path.join(output_directory, image.name + '_cropped.tif')
        print(f"Saving image: {save_file}\n")
        image.save_original_image_cropped(save_file)

        if output_thumbnails:
            save_thumbnails_path = os.path.join(output_directory, 'fiducial_thumbnails', image.name + '_thumbnail.tif')
            image.save_fiducial_data_thumbnail(save_thumbnails_path)


def process_image_set(images_directory, output_directory, x_crop, y_crop, output_thumbnails=False, n_threads=1):
    # load image paths
    image_filepaths = get_filepaths_from_directory(images_directory)

    # load fiducial templates
    fiducial_dirs = get_filepaths_from_directory('../../templates')
    fiducial_templates = []
    for fiducial_dir in fiducial_dirs:
        fiducial_templates.append(FiducialTemplate(fiducial_dir))

    # Create a partial function with the other arguments pre-filled
    process_func = partial(process_single_image, fiducial_templates=fiducial_templates, x_crop=x_crop,
                           y_crop=y_crop, output_directory=output_directory, output_thumbnails=output_thumbnails)

    with Pool(n_threads) as pool:
        with tqdm(total=len(image_filepaths), desc="Processing images") as pbar:
            for _ in pool.imap_unordered(process_func, image_filepaths):
                pbar.update()


def get_filepaths_from_directory(directory):
    return [os.path.join(directory, filename) for filename in os.listdir(directory)]


def find_fiducials(image, templates):
    results = {}
    region = None
    offset_x, offset_y = 0, 0

    for fiducial_template in templates:  # Outer loop over years
        # print(f"Processing fiducial template_image: {fiducial_template.name}")

        for fiducial in fiducial_template.fiducials:  # Inner loop over positions (top, bottom, etc.)

            img_height, img_width = image.shape[:2]

            # Determine the region to search based on the position key
            if fiducial.position == 'top':
                region = image[:img_height // 2, :]  # Top half of the image
                offset_x, offset_y = 0, 0

            elif fiducial.position == 'bottom':
                region = image[img_height // 2:, :]  # Bottom half of the image
                offset_x, offset_y = 0, img_height // 2

            elif fiducial.position == 'left':
                region = image[:, :img_width // 2]  # Left half of the image
                offset_x, offset_y = 0, 0

            elif fiducial.position == 'right':
                region = image[:, img_width // 2:]  # Right half of the image
                offset_x, offset_y = img_width // 2, 0

            # Debugging: Check the position key and template_image shape
            # print(f"Fiducial template: {fiducial_template.name}, Position: {fiducial.position}")
            # print(f"Template size (x, y): {fiducial.image.shape}")
            # Perform template_image matching in the selected region
            result = cv2.matchTemplate(region, fiducial.image, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Calculate the center point in the full image context
            template_width, template_height = fiducial.image.shape[::-1]
            center_x = max_loc[0] + template_width // 2 + offset_x
            center_y = max_loc[1] + template_height // 2 + offset_y

            # Store the results in the dictionary, organized by year and position
            if fiducial_template.name not in results:
                results[fiducial_template.name] = {}

            results[fiducial_template.name][fiducial.position] = {
                'center_x': center_x,
                'center_y': center_y,
                'match_score': max_val
            }

    return results


def get_top_scores(results, top_n=4):
    """
    Extracts the top N scores from the results dictionary, ensuring
    that all positions (top, bottom, left, right) are included.

    Parameters:
    - results: The nested dictionary containing match scores.
    - top_n: The number of top scores to return (default is 4).

    Returns:
    - A list of tuples containing (year, position, center_x, center_y, match_score)
    """
    try:
        # Flatten the results into a list of tuples (year, position, center_x, center_y, match_score)
        flattened_results = []
        for year, positions in results.items():
            for position, data in positions.items():
                flattened_results.append(
                    (year, position, data['center_x'], data['center_y'], data['match_score'])
                )

        # Sort the list by match_score in descending order
        sorted_results = sorted(flattened_results, key=lambda x: x[4], reverse=True)

        # Initialize containers to hold the top results
        top_results = []
        positions_found = set()

        # Collect the top results ensuring all positions are covered
        for result in sorted_results:
            try:
                year, position, center_x, center_y, match_score = result
                if position not in positions_found:
                    top_results.append(result)
                    positions_found.add(position)
                if len(top_results) == top_n:
                    break
            except Exception as e:
                print(f"Error processing result {result}: {e}")

        # If we don't have all 4 positions, keep adding the next best scores until we do
        if len(positions_found) < 4:
            for result in sorted_results[len(top_results):]:
                try:
                    year, position, center_x, center_y, match_score = result
                    if position not in positions_found:
                        top_results.append(result)
                        positions_found.add(position)
                    if len(top_results) >= 4 and len(positions_found) == 4:
                        break
                except Exception as e:
                    print(f"Error processing result {result}: {e}")

        return top_results

    except Exception as e:
        print(f"An error occurred while processing the results: {e}")
        return []  # Return an empty list if something goes wrong


def get_fiducial_points(top_results):
    """
    Extracts the relevant points for each fiducial marker based on the template search results.

    Parameters:
    - top_results: A list of tuples containing (year, position, center_x, center_y, match_score)

    Returns:
    - A dictionary with the relevant points for each fiducial marker:
      {
        "top": x_top,
        "bottom": x_bottom,
        "left": y_left,
        "right": y_right
      }
    """
    # Initialize variables for x and y values
    top = left = right = bottom = None

    points = []
    # Extract the necessary x and y values from the top results
    for year, position, center_x, center_y, match_score in top_results:
        if position == "top":
            top = Point(center_x, center_y)
        elif position == "right":
            right = Point(center_x, center_y)
        elif position == "left":
            left = Point(center_x, center_y)
        elif position == "bottom":
            bottom = Point(center_x, center_y)
    # Ensure all necessary values were found
    if bottom is None or right is None or left is None or bottom is None:
        raise ValueError("Not all necessary fiducial positions were found in the top results.")

    # Return the relevant points
    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right
    }


def compute_principal_point(fiducial_points):
    """
    Computes the center (principal) point based on the fiducial points.

    Parameters:
    - fiducial_points: A dictionary containing the relevant points for each fiducial marker:
      {
        "top": x_top,
        "bottom": x_bottom,
        "left": y_left,
        "right": y_right
      }

    Returns:
    - A tuple representing the principal point (center_x, center_y)
    """
    top = fiducial_points.get("top")
    bottom = fiducial_points.get("bottom")
    left = fiducial_points.get("left")
    right = fiducial_points.get("right")

    # Calculate the center of the quadrilateral by averaging the x and y coordinates
    center_x = round((top.x + bottom.x + left.x + right.x) / 4)
    center_y = round((top.y + bottom.y + left.y + right.y) / 4)

    return PrincipalPoint(center_x, center_y)
