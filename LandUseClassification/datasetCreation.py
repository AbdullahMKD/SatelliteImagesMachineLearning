import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Get the absolute path of the data directory
data_dir = os.path.abspath('Data')

# Check if the directory exists
if not os.path.exists(data_dir):
    print(f"Error: Directory '{data_dir}' does not exist.")
    exit()

# Check if there are any files in the directory
files = os.listdir(data_dir)
if not files:
    print(f"Error: Directory '{data_dir}' is empty.")
    exit()

# Try opening an image to check for corruption (optional)
try:
    # Open the first image
    img = cv2.imread(os.path.join(data_dir, files[0]))
    if img is None:
        print(f"Warning: cv2.imread() failed to open '{os.path.join(data_dir, files[0])}'.")
    del img  # Explicitly release the image (optional)
except Exception as e:
    print(f"Error: An error occurred while opening an image: {e}")
    exit()

datagen = ImageDataGenerator(rescale=1/5, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = datagen.flow_from_directory(
    data_dir,
    batch_size=56,
    class_mode='input',
    shuffle=True,
    seed=69,
)
print(os.path.abspath('Data'))
print(os.listdir('Data'))

for images in train_generator:
    print(images)
    break