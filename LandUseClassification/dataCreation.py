import cv2
import numpy as np
import pandas as pd

# Define the paths to your JPG images and label data
image_paths = ['Data/Lahore_2023-05-03-00_00_2023-06-03-23_59_Sentinel'
               '-2_L2A_True_color.jpg',
               "Data/Arg_Chile_Border_2023-01-01-00_00_2023-02-01-23_59_Sentinel"
               "-2_L2A_True_color.jpg",
               "Data/Al-Dhannah_2022-05-20-00_00_2022-06-20-23_59_Sentinel-2_L2A_True_color.jpg"]
print(len(image_paths))


def pixel_to_label(pixel_value):
    if pixel_value <= 50:
        return 1
    elif pixel_value <= 100:
        return 2
    elif pixel_value <= 150:
        return 3
    elif pixel_value <= 200:
        return 4
    else:
        return 5


# Define function to preprocess image
def preprocess_image(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (993, 647))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    return img


def process_and_store_data(images):
    img_pixel_data = []
    img_label_data = []
    # Preprocess image
    image = preprocess_image(images)
    # Flatten image to 1D array
    flattened_image = image.flatten()

    # Shuffle the flattened image pixels
    np.random.shuffle(flattened_image)
    # Generate labels and append data
    img_pixel_data.extend(flattened_image)
    labels = [pixel_to_label(pixel) for pixel in flattened_image]
    img_label_data.extend(labels)

    return img_pixel_data, img_label_data


all_pixel_data = []
all_label_data = []

for image_path in image_paths:
    pixel_data, label_data = process_and_store_data(image_path)
    all_pixel_data.extend(pixel_data)
    all_label_data.extend(label_data)

df = pd.DataFrame({'Pixels': all_pixel_data, 'labels': all_label_data})

df.to_csv('training_data.csv', index=False)
