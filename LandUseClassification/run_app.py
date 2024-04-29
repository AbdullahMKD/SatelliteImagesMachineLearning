import tkinter as tk
from gui.main_window import main_window
from processing.kmeans_processor import kMeans_processing as kMp
from processing.image_processing import image_processor as imgp


def main():
    root = tk.Tk()
    kmeans_processor = kMp()
    image_processor = imgp()
    app = main_window(root, kmeans_processor, image_processor)
    root.mainloop()


if __name__ == "__main__":
    main()
