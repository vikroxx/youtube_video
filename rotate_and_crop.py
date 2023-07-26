import cv2
import numpy as np
import math

def rotate_and_crop_d(image_path, angle):
    # Load the image
    img = cv2.imread(image_path)

    # Get image height and width
    h, w = img.shape[:2]
    angle_radians = math.radians(angle)
    
    # Calculate the new image dimensions
    rotated_width = int(h * abs(math.sin(angle_radians)) + w * abs(math.cos(angle_radians)))
    rotated_height = int(h * abs(math.cos(angle_radians)) + w * abs(math.sin(angle_radians)))

    # Create a rotation matrix (2x3) with the angle and scale (center of the rotation is the image center)
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)

    # Adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (rotated_width - w) / 2
    rotation_matrix[1, 2] += (rotated_height - h) / 2

    # Apply affine transformation on the image using the rotation matrix
    rotated_img = cv2.warpAffine(img, rotation_matrix, (rotated_width, rotated_height))

    # Calculate the size of the largest inscribed rectangle. We use the
    # absolute cos and sin to ensure that the values are positive.
    cosine = abs(math.cos(angle_radians))
    sine = abs(math.sin(angle_radians))
    inscribed_width = w * cosine - h * sine
    inscribed_height = h * cosine - w * sine

    # Calculate the position of the largest inscribed rectangle
    start_y = int((rotated_height - inscribed_height) / 2)
    start_x = int((rotated_width - inscribed_width) / 2)

    # Crop the image
    cropped_img = rotated_img[start_y:start_y + int(inscribed_height), start_x:start_x + int(inscribed_width)]

    return cropped_img


import cv2
import numpy as np
import math




# # Example usage:
# rotated_cropped_img = rotate_and_crop("test.jpg", 10)
# cv2.imwrite('rotated_and_cropped_image.jpg',rotated_cropped_img)

