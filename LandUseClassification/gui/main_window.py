from tkinter import filedialog, Label, Button, Entry, Canvas, Scrollbar, Frame, messagebox
from PIL import Image, ImageTk
from tkinter import Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from functools import partial
import tkinter as tk
import tkinter.ttk as ttk
import math


# noinspection PyTypeChecker
class main_window:
    def __init__(self, master, km_processor, img_processor):

        self.master = master
        self.km_processor = km_processor
        self.img_processor = img_processor
        self.master.title("Land Use Classification Using K-means Clustering")
        self.master.geometry("993x750")

        # Class attributes
        self.file_paths = []
        self.image_labels = []
        self.cluster_imgs = []
        self.menu_var1 = None
        self.menu_var2 = None

        # Canvas for scrollable image display
        self.image_display_frame = Frame(master)
        self.image_display_frame.pack(fill='both', expand=True)

        self.canvas = Canvas(self.image_display_frame, borderwidth=0)
        self.frame_images = Frame(self.canvas)
        self.vsb = Scrollbar(self.image_display_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((4, 4), window=self.frame_images, anchor="nw", tags="self.frame")

        self.frame_images.bind("<Configure>", lambda event, canvas=self.canvas: self.on_frame_configure())

        # Controls
        # Main frame for controls
        controls_frame = Frame(master, bd=2, relief="groove")
        controls_frame.pack(side='top', fill='x', padx=10, pady=10)

        self.entry_label = Label(controls_frame, text="Please enter a value of K you want to classify.")
        self.entry_label.grid(row=0, column=0, sticky='w', padx=5, pady=5)

        self.k_entry = Entry(controls_frame, width=10)
        self.k_entry.grid(row=0, column=1, padx=5, pady=5)

        self.load_button = Button(controls_frame, text="Load Images", command=self.load_images)
        self.load_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        self.start_button = Button(controls_frame, text="Start Classification", command=self.start_classification)
        self.start_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

    def load_images(self):
        for label in self.image_labels:
            label.destroy()
        self.image_labels.clear()

        file_paths = filedialog.askopenfilenames(title="Select images",
                                                 filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        self.file_paths = file_paths
        if not file_paths:
            return
        if file_paths:
            for path in file_paths:
                img = Image.open(path)
                img = img.resize((993, 647))
                img_tk = ImageTk.PhotoImage(img)
                label = Label(self.frame_images, image=img_tk)
                label.image = img_tk
                label.pack()
                self.image_labels.append(label)

    def start_classification(self):
        if self.file_paths:
            k_value = self.k_entry.get()
            if k_value.isdigit() and int(k_value) > 1:
                messagebox.showinfo("Starting Classification", "Please wait after closing the window")
                results_window = Toplevel(self.master)
                results_window.title("Classification Results")
                results_window.geometry("800x800")
                k = int(k_value)
                self.display_results(results_window, k)
            else:
                messagebox.showerror("Invalid Input", "Please enter a valid integer value for K.")
        else:
            messagebox.showerror("No Images loaded", "Please load images first.")

    def get_image_paths(self):
        return self.file_paths

    def on_frame_configure(self):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def display_results(self, results_window, k):
        num_images = len(self.file_paths)
        cols = 1
        rows = math.ceil(num_images / cols)
        fig = Figure(figsize=(5 * cols, 4 * rows), dpi=100)
        for index, path in enumerate(self.file_paths):
            self.cluster_imgs.append(self.km_processor.clustering(k, self.img_processor.process_data(path)))
            mapping = self.km_processor.get_mapping()
            self.cluster_imgs[-1] = self.km_processor.apply_mapping_to_labels(mapping)
            print("clustering done for: {}".format(path))
            metrics = {'Inertia': self.km_processor.get_inertia(),
                       'Silhouette score': self.km_processor.get_silhouette_score()}
            plot = fig.add_subplot(rows, cols, index + 1)
            plot.imshow(self.cluster_imgs[-1], cmap="viridis", vmin=0, vmax=k-1)
            plot.set_title(f"Image {index + 1}")
            plot.axis('off')
            metric_text = '\n'.join([f"{key}: {value:.2f}" if isinstance(value, (float, int)) else f"{key}: {value}"
                                     for key, value in metrics.items()])
            plot.text(1.05, 0.5, metric_text, transform=plot.transAxes, fontsize=9, verticalalignment='center')
        canvas = FigureCanvasTkAgg(fig, master=results_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        if num_images > 1:
            side_menu = tk.Frame(results_window)
            side_menu.pack(side="right", fill="y")
            self.menu_var1 = tk.IntVar()
            self.menu_var2 = tk.IntVar()
            menu1 = ttk.Combobox(side_menu, textvariable=self.menu_var1, values=list(range(num_images)))
            menu2 = ttk.Combobox(side_menu, textvariable=self.menu_var2, values=list(range(num_images)))
            menu1.pack()
            menu2.pack()
            select_button = tk.Button(side_menu, text="Create Difference Map",
                                      command=partial(self.create_diff_map))
            select_button.pack()

    def create_diff_map(self):
        index1 = self.menu_var1.get()
        index2 = self.menu_var2.get()
        if index1 != index2 and index1 < len(self.cluster_imgs) and index2 < len(self.cluster_imgs):
            index1 = self.cluster_imgs[index1]
            index2 = self.cluster_imgs[index2]
            difference = abs(index1 - index2)
            diff_window = Toplevel(self.master)
            diff_window.title("Difference Map")
            diff_window.geometry("993x800")
            fig = Figure(figsize=(5, 4), dpi=100)
            plot = fig.add_subplot(1, 1, 1)
            plot.imshow(difference, cmap='Spectral')
            plot.set_title("Difference Map")
            canvas = FigureCanvasTkAgg(fig, master=diff_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        else:
            messagebox.showerror("Invalid Selection", "Please select 2 different images.")
