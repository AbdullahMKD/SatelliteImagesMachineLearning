import cv2


class image_processor:
    def __init__(self):
        self.data = None

    def process_data(self, image_path):
        # Preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Image at {img} could not be loaded. Check the file path and format.")
        img = cv2.resize(img, (993, 647))
        img = img / 255.0
        print(img.shape)
        # Flatten image to 1D array
        flattened_image = img.flatten()
        if len(flattened_image) % 3 != 0:
            raise ValueError(
                "Flattened image data is not a multiple of 3, check image format and processing steps.")
        # Reshape data for Machine learning Algorithm
        self.data = flattened_image.reshape(-1, 3)
        return self.data

    def get_data(self):
        return self.data
