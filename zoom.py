import random
import os
import cv2
import numpy as np
import math
import sys
from tqdm import tqdm


def create_zoom_out_video(image_path, output_video_path, video_duration, zoom_percent, fps, frame_size=(1920, 1080)):
    image = cv2.imread(image_path)
    original_height, original_width, _ = image.shape
    print(image.shape)
    
    # Aspect ratio of the original image
    image_aspect_ratio = original_width / original_height

    # Aspect ratio of the video frame
    frame_aspect_ratio = frame_size[0] / frame_size[1]

    # Determine cropping dimensions to maintain aspect ratio
    if image_aspect_ratio > frame_aspect_ratio:
        new_width = int(original_height * frame_aspect_ratio)
        # updated_frame_size = (int(frame_size[0]*original_height/1080), int(frame_size[1]*original_height/1080))
        start_x = int((original_width - new_width) / 2)
        updated_frame_size = (new_width, original_height)
    else:
        new_height = int(original_width / frame_aspect_ratio)
        # updated_frame_size = (int(frame_size[0]*original_width/1920), int(frame_size[1]*original_width/1920))
        start_y = int((original_height - new_height) / 2)
        updated_frame_size = (original_width, new_height)

    print(updated_frame_size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    total_frames = video_duration * fps
    raw_zoom_step_size = ((zoom_percent / 100) * min(updated_frame_size)) / ( total_frames)

    if raw_zoom_step_size < 1:
        zoom_interval = int(1 / raw_zoom_step_size)
        zoom_step_size = 1
    else:
        zoom_interval = 1
        zoom_step_size = int(raw_zoom_step_size)
    print('raw_zoom_step_size : ', raw_zoom_step_size)
    print('Zoom interval : ', zoom_interval)
    zoom_counter = 0

    start_y = int((zoom_percent / 100) * updated_frame_size[1])
    if raw_zoom_step_size <1 :
        zoom_interval +=1 
    for frame_num in range(total_frames):

        if zoom_counter % zoom_interval == 0:
            start_y = max(0,start_y - zoom_step_size)
            end_y = updated_frame_size[1] - start_y

            # start_y = (total_frames - frame_num) * zoom_step_size
            # end_y = updated_frame_size[1] - ((total_frames - frame_num) * zoom_step_size)

            # print(start_y, end_y, end_y-start_y)
            # Calculate the x boundaries to maintain aspect ratio
            height = end_y - start_y
            width = int(height * frame_aspect_ratio)
            start_x = (updated_frame_size[0] - width) // 2
            end_x = start_x + width

            cropped_frame = image[start_y:end_y, start_x:end_x]
        else:
            # If not zooming this frame, simply copy the previous frame
            cropped_frame = cropped_frame.copy()

        final_frame = cv2.resize(cropped_frame, frame_size)

        video.write(final_frame)
        zoom_counter += 1

    cv2.destroyAllWindows()
    video.release()


def create_zoom_video(image_path, output_video_path, video_duration, zoom_percent, fps, frame_size=(1920, 1080)):
    image = cv2.imread(image_path)
    original_height, original_width, _ = image.shape

    # Aspect ratio of the original image
    image_aspect_ratio = original_width / original_height

    # Aspect ratio of the video frame
    frame_aspect_ratio = frame_size[0] / frame_size[1]

    # Determine cropping dimensions to maintain aspect ratio
    if image_aspect_ratio > frame_aspect_ratio:
        new_width = int(original_height * frame_aspect_ratio)
        start_x = int((original_width - new_width) / 2)
        updated_frame_size = (new_width, original_height)
    else:
        new_height = int(original_width / frame_aspect_ratio)
        start_y = int((original_height - new_height) / 2)
        updated_frame_size = (original_width, new_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    total_frames = video_duration * fps
    raw_zoom_step_size = ((zoom_percent / 100) * min(updated_frame_size)) / total_frames

    if raw_zoom_step_size < 1:
        zoom_interval = int(1 / raw_zoom_step_size)
        zoom_step_size = 1
    else:
        zoom_interval = 1
        zoom_step_size = int(raw_zoom_step_size)

    zoom_counter = 0
    print('raw_zoom_step_size : ', raw_zoom_step_size)
    if raw_zoom_step_size < 1:
        zoom_interval +=1
    start_y =0 
    for frame_num in range(total_frames):
        if zoom_counter % zoom_interval == 0:
            # start_y = min(frame_num * zoom_step_size, (updated_frame_size[1] - 1) // 2)
            # end_y = max(updated_frame_size[1] - frame_num * zoom_step_size, (updated_frame_size[1] + 1) // 2)
            
            start_y = start_y + zoom_step_size
            end_y = updated_frame_size[1] - start_y

            # print(start_y, end_y, end_y - start_y)
            # Calculate the x boundaries to maintain aspect ratio
            height = end_y - start_y
            width = int(height * frame_aspect_ratio)
            start_x = (updated_frame_size[0] - width) // 2
            end_x = start_x + width

            cropped_frame = image[start_y:end_y, start_x:end_x]
        else:
            # If not zooming this frame, simply copy the previous frame
            cropped_frame = cropped_frame.copy()

        final_frame = cv2.resize(cropped_frame, frame_size)

        video.write(final_frame)
        zoom_counter += 1

    cv2.destroyAllWindows()
    video.release()


def create_pan_video(image_path, output_video_path, video_duration, zoom_percent, pan_direction, fps, frame_size=(1920, 1080)):
    image = cv2.imread(image_path)
    original_height, original_width, _ = image.shape

    # Aspect ratio of the original image
    image_aspect_ratio = original_width / original_height

    # Aspect ratio of the video frame
    frame_aspect_ratio = frame_size[0] / frame_size[1]
    print('frame aspect ratio' ,frame_aspect_ratio)
    # Determine cropping dimensions to maintain aspect ratio
    if image_aspect_ratio > frame_aspect_ratio:
        new_width = int(original_height * frame_aspect_ratio)
        updated_frame_size = (new_width, original_height)
    else:
        new_height = int(original_width / frame_aspect_ratio)
        updated_frame_size = (original_width, new_height)

    # Calculate vertical start and end indices
    start_y = (original_height - int(((100 - zoom_percent) / 100) * updated_frame_size[1])) // 2
    end_y = start_y + int(((100- zoom_percent) / 100) * updated_frame_size[1])
    # print(end_y-start_y)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    total_frames = video_duration * fps

    # Calculate the maximum amount of panning based on the zoom level
    pan_range = int((zoom_percent / 100) * updated_frame_size[0])

    if pan_direction == 'left':
        start_x = pan_range
    elif pan_direction == 'right':
        start_x = 0

    # How many pixels to pan per frame
    raw_pan_step_size =  pan_range / total_frames

    if raw_pan_step_size < 1:
        pan_interval = int(1 / raw_pan_step_size)
        pan_step_size = 1
    else:
        pan_interval = 1
        pan_step_size = int(raw_pan_step_size)

    print('raw_pan_step_size : ',raw_pan_step_size)
    
    if raw_pan_step_size < 1:
        pan_interval +=1
    for frame_num in range(total_frames):
        end_x = start_x + updated_frame_size[0] - pan_range
        cropped_frame = image[start_y:end_y, start_x:end_x]
        # if frame_num ==0:
        #     cv2.imshow('window', cropped_frame)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows
        final_frame = cv2.resize(cropped_frame, frame_size)
        # if frame_num ==0:
        #     cv2.imshow('window', final_frame)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows
        if frame_num % pan_interval == 0:
            if pan_direction == 'left':
                # start_x = max(0, start_x - pan_step_size)
                start_x =start_x - pan_step_size

            elif pan_direction == 'right':
                # start_x = min(pan_range, start_x + pan_step_size)
                start_x = start_x + pan_step_size
        # print(start_x, end_x, end_x - start_x)
        video.write(final_frame)

    cv2.destroyAllWindows()
    video.release()


def create_pendulum_video(image_path, output_video_path, video_duration, zoom_percent, num_cycles, amplitude_factor, fps, frame_size=(1920, 1080)):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Aspect ratio of the original image
    image_aspect_ratio = width / height

    # Aspect ratio of the video frame
    frame_aspect_ratio = frame_size[0] / frame_size[1]

    # Resize image such that it fills the frame
    if image_aspect_ratio > frame_aspect_ratio:
        new_width = int(height * frame_aspect_ratio)
        start_x = int((width - new_width) / 2)
        # cropped_image = image[:, start_x:start_x+new_width]
        updated_frame_size = (new_width, height)

    else:
        new_height = int(width / frame_aspect_ratio)
        start_y = int((height - new_height) / 2)
        # cropped_image = image[start_y:start_y+new_height, :]
        updated_frame_size = (width, new_height)

    # Calculate vertical start and end indices
    start_y = (height - int(((100 - zoom_percent) / 100) * updated_frame_size[1])) // 2
    end_y = start_y + int(((100- zoom_percent) / 100) * updated_frame_size[1])

    # Resize the cropped image to match the frame size
    # resized_image = cv2.resize(cropped_image, frame_size)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    total_frames = video_duration * fps

    # Calculate the maximum amount of panning based on the zoom level
    max_pan_range = int(((zoom_percent / 100) * updated_frame_size[0]) / 2)
    # amplitude_factor determines the range of the pendulum motion
    pan_range = int(amplitude_factor * max_pan_range)
    # print(amplitude_factor)
    for frame_num in range(total_frames):
        # Simulate the pendulum motion with a sine function
        pendulum_pos = pan_range * \
            math.sin(2 * math.pi * num_cycles * frame_num / total_frames)

        start_x = max_pan_range + int(pendulum_pos)
        end_x = start_x + updated_frame_size[0] - 2 * max_pan_range
        # print(start_x, end_x, end_x - start_x)
        cropped_frame = image[start_y:end_y, start_x:end_x]
        final_frame = cv2.resize(cropped_frame, frame_size)

        video.write(final_frame)
        # print(start_y, end_y)

    cv2.destroyAllWindows()
    video.release()



def rotate_and_crop(cv2_image, angle):
    # Load the image
    img = cv2_image

    # Get image height and width
    h, w = img.shape[:2]
    aspect_ratio = w / h
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

    # Crop the image to the largest inscribed rectangle
    cropped_img = rotated_img[start_y:start_y + int(inscribed_height), start_x:start_x + int(inscribed_width)]

    # Adjust the final cropping to maintain the original aspect ratio
    final_height, final_width = cropped_img.shape[:2]
    if final_width / final_height > aspect_ratio:
        # If the width is too big, adjust it
        final_width = int(final_height * aspect_ratio)
    else:
        # If the height is too big, adjust it
        final_height = int(final_width / aspect_ratio)
        
    # Calculate the position of the final cropped rectangle
    start_y = (cropped_img.shape[0] - final_height) // 2
    start_x = (cropped_img.shape[1] - final_width) // 2

    # Final cropping
    final_cropped_img = cropped_img[start_y:start_y + final_height, start_x:start_x + final_width]
    
    return final_cropped_img

def create_zoom_out_video_with_rotation(image_path, output_video_path, video_duration, zoom_percent, fps, frame_size=(1920, 1080), start_rotation_angle=15):
    image = cv2.imread(image_path)
    original_height, original_width, _ = image.shape

    # Aspect ratio of the original image
    image_aspect_ratio = original_width / original_height

    # Aspect ratio of the video frame
    frame_aspect_ratio = frame_size[0] / frame_size[1]

    # Determine cropping dimensions to maintain aspect ratio
    if image_aspect_ratio > frame_aspect_ratio:
        new_width = int(original_height * frame_aspect_ratio)
        start_x = int((original_width - new_width) / 2)
        updated_frame_size = (new_width, original_height)
    else:
        new_height = int(original_width / frame_aspect_ratio)
        start_y = int((original_height - new_height) / 2)
        updated_frame_size = (original_width, new_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    total_frames = video_duration * fps
    raw_zoom_step_size = ((zoom_percent / 100) * min(updated_frame_size)) / (total_frames)

    if raw_zoom_step_size < 1:
        zoom_interval = int(1 / raw_zoom_step_size)
        zoom_step_size = 1
    else:
        zoom_interval = 1
        zoom_step_size = int(raw_zoom_step_size)
    
    start_y = int((zoom_percent / 100) * updated_frame_size[1])
    if raw_zoom_step_size < 1:
        zoom_interval += 1

    # Initialize rotation
    rotation_step = start_rotation_angle / total_frames
    current_angle = start_rotation_angle

    for frame_num in tqdm(range(total_frames)):
        if frame_num % zoom_interval == 0:
            start_y = max(0, start_y - zoom_step_size)
            end_y = updated_frame_size[1] - start_y

            height = end_y - start_y
            width = int(height * frame_aspect_ratio)
            start_x = (updated_frame_size[0] - width) // 2
            end_x = start_x + width

            cropped_frame = image[start_y:end_y, start_x:end_x]
        else:
            cropped_frame = cropped_frame.copy()

        rotated_frame = rotate_and_crop(cropped_frame, current_angle)
        # # Get rotation matrix and apply rotation
        # M = cv2.getRotationMatrix2D((frame_size[0] / 2, frame_size[1] / 2), current_angle, 1)
        # rotated_frame = cv2.warpAffine(cropped_frame, M, frame_size)

        # Resize the frame to eliminate padding caused by rotation

        resized_frame = cv2.resize(rotated_frame, frame_size)

        video.write(resized_frame)
        
        # Update angle for next rotation
        current_angle -= rotation_step

    cv2.destroyAllWindows()
    video.release()


# Use the function
# create_pendulum_video("test.jpg", 5, 15, 1, 0.6, 90)


# # Use the function
# create_pan_video("test.jpg", 5, 20, 'left', 40)


# create_zoom_video(r"images/1.png",'zoom_test.mp4', 4, 30, 30)
# create_zoom_out_video(image_path=r"images/1.png", output_video_path='zoom_test.mp4',
#                         video_duration=4, zoom_percent=30, fps=60, frame_size= (1920, 1080))

# create_zoom_out_video("test.jpg", 6, 35, 30)

# create_acceleated_zoom_video("images\\2.png", 5, 30, 60,0)
# create_accelerated_zoom_video_2("test.jpg", 6,35,60,0)

def random_function_calling(image_path,
                            output_video_path,
                            FPS=45,
                            zoom_level=int(random.random() * 5 + 10),
                            time_duration=7,
                            # num_oscillations=1,
                            # amplitude_factor=round(
                            #     random.random()*0.7 + 0.5, 2),
                            direction=random.choice(['left', 'right']),
                            rotation_angle = -15 + int(random.random()*30),
                            frame_size= (1920, 1080)):
    print(zoom_level)
    random_number = random.randint(0, 3)
    # print('image : {} \nrandom_number  : {}'.format(image_path, random_number))
    # print(image_path, output_video_path, time_duration, zoom_level, FPS)
    # create_pan_video(image_path=image_path, output_video_path=output_video_path,pan_direction= direction, 
    #                       video_duration=5, zoom_percent=20, fps=40, frame_size=frame_size)
    # create_pendulum_video(image_path=image_path, output_video_path=output_video_path,
    #                           video_duration=time_duration, zoom_percent=zoom_level, num_cycles=num_oscillations, 
    #                           amplitude_factor=amplitude_factor, fps=FPS, frame_size=frame_size)
    # sys.exit()
    if random_number == 0:
        create_zoom_video(image_path=image_path, output_video_path=output_video_path,
                          video_duration=time_duration, zoom_percent=zoom_level, fps=FPS, frame_size=frame_size)

    elif random_number == 1:
        create_zoom_out_video(image_path=image_path, output_video_path=output_video_path,
                              video_duration=time_duration, zoom_percent=zoom_level, fps=FPS, frame_size=frame_size)

    elif random_number == 2:
        create_pan_video(image_path=image_path, output_video_path=output_video_path, video_duration=time_duration,
                         zoom_percent=zoom_level, pan_direction=direction, fps=FPS, frame_size=frame_size)

    elif random_number == 3:
        create_zoom_out_video_with_rotation(image_path=image_path, output_video_path=output_video_path,
                              video_duration=time_duration, zoom_percent=zoom_level, start_rotation_angle= rotation_angle, 
                              fps=FPS, frame_size=frame_size)


# print(random.choice(['left','right']))
# print(int(random.random() * 25 + 10))

files = os.listdir('images')

if not os.path.isdir('video'):
    os.mkdir('video')

frame_size = (1920, 1080)
for file in files:
    video_path  = os.path.join('video' , '.'.join(os.path.basename(file).split('.')[:-1])+'.mp4')
    random_function_calling(image_path= os.path.join('images', file),
                            output_video_path= video_path)

# create_zoom_out_video_with_rotation(image_path=r'images\1_clipdrop-enhance.png', output_video_path='test3.mp4',video_duration=6, zoom_percent=13
#                                     , fps=40, frame_size=(1920, 1080), start_rotation_angle= -5)