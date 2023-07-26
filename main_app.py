import os
import cv2
import numpy as np
from scipy.ndimage import binary_dilation
from tqdm import tqdm
import random

images = os.listdir('images')
images = sorted(images, key= lambda x: int(x.split('.')[0]))
print(images)

from scipy.ndimage import binary_dilation

def get_boundary_pixels(mask):
    # Dilate the mask to get the boundary
    dilated_mask = binary_dilation(mask)
    # The boundary is the difference between the dilated mask and the original mask
    boundary = dilated_mask ^ mask
    # Get the coordinates of the boundary pixels
    boundary_pixels = np.argwhere(boundary == 1)
    return boundary_pixels

def get_neighbors_not_in_mask(mask, pixel):
    y, x = pixel
    neighbors = [(ny, nx) for nx in range(x-1, x+2) for ny in range(y-1, y+2) 
                 if (ny, nx) != (y, x) and 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]]
    return [(ny, nx) for (ny, nx) in neighbors if mask[ny, nx] == 0]

def update_mask(mask, num_pixels_to_add):
    boundary_pixels = get_boundary_pixels(mask)
    old_mask = mask.copy()  # Keep a copy of the current mask state

    new_pixels = []
    
    if len(boundary_pixels) == 0:
        return mask, False  # No more boundary pixels left, end the loop

    for pixel in boundary_pixels:
        neighbors_not_in_mask = get_neighbors_not_in_mask(mask, pixel)
        if neighbors_not_in_mask:
            # Select a number of neighbors that is not in the mask
            new_pixels += random.sample(neighbors_not_in_mask, min(num_pixels_to_add, len(neighbors_not_in_mask)))

    # Add the new pixels to the mask
    for pixel in new_pixels:
        y, x = pixel
        mask[y, x] = 1

    if np.array_equal(mask, old_mask):
        return mask, False

    flag = True
    if np.size(mask) == np.count_nonzero(mask):
        flag = False

    return mask, flag

# def update_mask(mask):
#     boundary_pixels = get_boundary_pixels(mask)
#     # To store newly added boundary pixels
#     new_pixels = []

#     for pixel in boundary_pixels:
#         neighbors_not_in_mask = get_neighbors_not_in_mask(mask, pixel)
#         if neighbors_not_in_mask:
#             # Select randomly one of the neighbors that is not in the mask
#             new_pixel = random.choice(neighbors_not_in_mask)
#             new_pixels.append(new_pixel)

#     # Add the new pixels to the mask
#     for pixel in new_pixels:
#         y, x = pixel
#         mask[y, x] = 1

#     # print("Total number of pixels in mask:", np.size(mask))
#     # print("Number of 1s in mask:", np.count_nonzero(mask))
#     flag = True
#     if np.size(mask) == np.count_nonzero(mask):
#         flag = False

#     return mask, flag



# Function to create a transition video between two images
def create_transition(image1, image2, pixel_cluster_size, output_path):

    height, width, _ = image1.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'

    # Initialize mask at a random position
    mask = np.zeros((height, width), np.uint8)
    starting_point = (np.random.randint(height), np.random.randint(width))
    mask[starting_point] = 1

    continue_mask = True
    frames = []
    counter = 0
    while continue_mask:
        for _ in range(pixel_cluster_size):
            mask, continue_mask = update_mask(mask, num_pixels_to_add= 1000)
            mask_pixels = np.sum(mask)
            print(f"Mask covers {mask_pixels}/{width * height}")

            if not continue_mask:
                break
        frame = image1 * np.stack([mask]*3, axis=-1) + image2 * np.stack([(1-mask)]*3, axis=-1)
        # if counter%100 == 0:
        #     cv2.imshow('Frame', frame)
        #     cv2.waitKey(1)  
        frames.append(frame)

    # video.write(np.uint8(frame))


    length = len(frames)
    final_frames = []
    time_duration = 0.5
    fps = 30
    total_frames = int(fps * time_duration)
    print(length, total_frames)
    if length > total_frames:
        for i in range(total_frames -1):
            final_frames.append(frames[i * length//total_frames])
        final_frames.append(frames[-1])
    
    frames = final_frames   
    
    
    # fps = int(len(frames)/10)
    print('FPS : {}'.format(fps))
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # video = cv2.VideoWriter('transition.mp4', fourcc, 30, (img1.width, img1.height))
    import pickle
    pickle.dump(frames, open('frames.p','wb'))
    
    for frame in tqdm(frames):
        video.write(np.uint8(frame))
    video.release()

# Use it like this:
image1 = cv2.imread('images//1.png')
image2 = cv2.imread('images//2.png')
create_transition(image1, image2, pixel_cluster_size=5, output_path='transition.mp4')
