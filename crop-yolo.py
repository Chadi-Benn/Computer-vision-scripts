
import os
import numpy as np
import files_mgmt
import cv2
import matplotlib.pyplot as plt


path_img = '/home/chadi/Pictures/controle_grille/C'
path_label = '/home/chadi/yolov5_test/runs/detect/exp14/labels'



label_list = files_mgmt.list_labels(path_label)[0]

# Define a new function that returns a new list containing couples of the respective elements from the two lists

def create_list(list_b, list_c):
    list_a = []

    # Check if the lengths of list_b and list_c are the same
    if len(list_b) != len(list_c):
        raise ValueError("list_b and list_c must have the same length.")

    for i in range(len(list_b)):
        list_a.append([list_b[i], list_c[i]])

    return list_a

# Define a new function that extracts the bnd box coordinates from the label file

def extract_coordinates(filename):
    word_list = []
    
    with open(filename, 'r') as file:
        for line in file:
            words = line.strip().split()
            if len(words) >= 3:
                word_tuple = (np.double(words[1]), np.double(words[2]))              
                word_list.append(word_tuple)
    
    return word_list

# Define the crop function

def crop_image(image, center_x, center_y, desired_box_width, desired_box_height):
    # Get image size
    image_height, image_width = image.shape[:2]

    # Convert center coordinates and box dimensions to absolute values
    abs_center_x = int(center_x * image_width)
    abs_center_y = int(center_y * image_height)

    # Calculate the top-left corner of the bounding box
    x1 = abs_center_x - (desired_box_width // 2)
    y1 = abs_center_y - (desired_box_height // 2)

    # Calculate the bottom-right corner of the bounding box
    x2 = x1 + desired_box_width
    y2 = y1 + desired_box_height

    # Adjust x1 and x2 if they are out of bounds
    if x1 < 0:
        x2 -= x1
        x1 = 0

    if x2 > image_width:
        x1 -= (x2 - image_width)
        x2 = image_width

    # Adjust y1 and y2 if they are out of bounds
    if y1 < 0:
        y2 -= y1
        y1 = 0

    if y2 > image_height:
        y1 -= (y2 - image_height)
        y2 = image_height

    # Crop the image using the adjusted bounding box coordinates
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image

# Extract the coordinates from the label files

all_coordinates_list = []
for i in label_list : 
    all_coordinates_list.append(extract_coordinates(i))

# Create a list of all the images paths

img_list = files_mgmt.list_raw_img(path_img)[0]

# Create a list of (image_path,centre coordinates)
img_bbox_coor = create_list(img_list,all_coordinates_list)



cptr = 0
cropped_path = '/home/chadi/Pictures/controle_grille/Cropped/C'

# Change directory in order to save the new images
os.makedirs(cropped_path, exist_ok=True)


for i in img_bbox_coor:
    img = cv2.imread(i[0])

    #Crop the new bounding box with the desired size 
    
    cropped_bnd_box = crop_image(img, i[1][0][0], i[1][0][1], 150, 150) 

    # Define the new image path

    save_path = os.path.join(cropped_path, str(cptr) + '.png')
    cv2.imwrite(save_path, cropped_bnd_box)
    cptr += 1

   


print("Done")