import cv2
import numpy as np
import pandas as pd

# Define the paths to your JPG images and label data
image_paths = ['../Data/Lahore_2023-05-03-00_00_2023-06-03-23_59_Sentinel'
               '-2_L2A_True_color.jpg',
               "../Data/Arg_Chile_Border_2023-01-01-00_00_2023-02-01-23_59_Sentinel"
               "-2_L2A_True_color.jpg",
               "../Data/Al-Dhannah_2022-05-20-00_00_2022-06-20-23_59_Sentinel-2_L2A_True_color.jpg"]
print(len(image_paths))


# Function to preprocess image
def preprocess_image(img):
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image at {img} could not be loaded. Check the file path and format.")
    img = cv2.resize(img, (993, 647))
    img = img / 255.0
    print(img.shape)
    return img


def process_and_store_data(images):
    img_pixel_data = []
    # Preprocess image
    image = preprocess_image(images)
    # Flatten image to 1D array
    flattened_image = image.flatten()
    if len(flattened_image) % 3 != 0:
        raise ValueError("Flattened image data is not a multiple of 3, check image format and processing steps.")
    # Generate labels and append data
    img_pixel_data.extend(flattened_image)
    return img_pixel_data


all_pixel_data = []

for image_path in image_paths:
    try:
        pixel_data = process_and_store_data(image_path)
        print(f"Processed {len(pixel_data) // 3} pixels from {image_path}")
        all_pixel_data.extend(pixel_data)
    except ValueError as e:
        print(e)
        continue

if len(all_pixel_data) % 3 != 0:
    raise ValueError("Total pixel data is not properly aligned to represent RGB values.")


reshaped_pixel_data = np.reshape(all_pixel_data, (-1, 3))
df = pd.DataFrame(reshaped_pixel_data, columns=['Pixel_R', 'Pixel_G', 'Pixel_B'])
print(df)

df.to_csv('training_data.csv', index=False)
