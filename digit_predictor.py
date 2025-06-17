import os
import sys  # Import sys
import numpy as np
import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow import keras
from tkinter import messagebox, Scale, HORIZONTAL
from PIL import Image, ImageDraw, ImageTk, ImageOps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import webbrowser
import ctypes # For Windows taskbar icon fix

# --- Configuration ---
# In a bundled app, Keras/TF will often find it if it's in the same directory.
MODEL_PATH = "best_model.keras" # Change to relative path for bundled app

CANVAS_SIZE = 280
PREDICTION_IMAGE_SIZE = 28
CONFIDENCE_THRESHOLD = 0.7
BACKGROUND_COLOR = "#f0f0f0"
BUTTON_COLOR = "#4CAF50"
CLEAR_COLOR = "#f44336"
SAVE_COLOR = "#FF9800"
ABOUT_COLOR = "#2196F3"

# Guide box parameters
GUIDE_BOX_WIDTH = int(CANVAS_SIZE * 0.8)
GUIDE_BOX_HEIGHT = int(CANVAS_SIZE * 0.8)
GUIDE_BOX_START_X = (CANVAS_SIZE - GUIDE_BOX_WIDTH) // 2
GUIDE_BOX_START_Y = (CANVAS_SIZE - GUIDE_BOX_HEIGHT) // 2
GUIDE_BOX_END_X = GUIDE_BOX_START_X + GUIDE_BOX_WIDTH
GUIDE_BOX_END_Y = GUIDE_BOX_START_Y + GUIDE_BOX_HEIGHT


class DigitPredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Digit Recognizer")
        master.geometry("1000x750")
        master.configure(bg=BACKGROUND_COLOR)
        
        # --- Handle icon path for PyInstaller bundling ---
        # Get the base directory for resources
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            # Running as a PyInstaller bundled executable
            basedir = sys._MEIPASS
        else:
            # Running as a regular Python script
            basedir = os.path.dirname(__file__)

        # Construct the path to the icon file
        # PyInstaller's --add-data will place 'digit_icon.ico' directly in 'basedir'
        self.icon_file = os.path.join(basedir, 'digit_icon.ico')

        # Set window icon
        try:
            # Tkinter's iconbitmap requires a .ico file and handles paths for bundled apps
            master.iconbitmap(default=self.icon_file)
        except tk.TclError:
            print(f"Warning: Could not load Tkinter window icon from {self.icon_file}.")
            # This 'pass' means if the icon isn't found, the app still runs without it.
            # Consider adding a more robust fallback or visual indicator.
            pass

        # Set AppUserModelID for Windows taskbar icon grouping (important for win7+)
        if sys.platform.startswith('win'):
            try:
                # Use a unique ID for your application
                myappid = 'MyCompany.DigitRecognizer.1_0'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except AttributeError:
                # This can happen if not running on Windows or if ctypes fails for some reason
                print("Warning: Could not set AppUserModelID (Windows taskbar grouping).")
                pass

        # Make window appear in taskbar (this code is mostly for initial window behavior, not the icon itself)
        master.attributes('-topmost', True)
        master.after_idle(master.attributes, '-topmost', False)

        # Try to load the model
        # Adjust MODEL_PATH here to reflect its location in the bundled app
        # If your model is in C:\... during development, but bundled to the root,
        # you need to ensure the bundled path is used.
        # For a bundled app, PyInstaller should put 'best_model.keras' in 'basedir'
        bundled_model_path = os.path.join(basedir, "best_model.keras")
        try:
            self.model = keras.models.load_model(bundled_model_path)
            print(f"Model loaded successfully from {bundled_model_path}")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model:\n{e}\n\nExpected model at: {bundled_model_path}")
            self.model = None

        self.last_x, self.last_y = None, None
        self.drawing_line_width = 20
        self.drawing_out_of_bounds = False

        # --- Header ---
        self.header = tk.Label(master, text="MNIST Digit Recognizer",
                               font=("Helvetica", 18, "bold"), bg=BACKGROUND_COLOR)
        self.header.pack(pady=10)

        # --- Main Content Frame (Horizontal Layout) ---
        self.main_frame = tk.Frame(master, bg=BACKGROUND_COLOR)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Left Column (Drawing Canvas) ---
        self.left_column = tk.Frame(self.main_frame, bg=BACKGROUND_COLOR, width=400)
        self.left_column.pack(side=tk.LEFT, fill=tk.BOTH, padx=20, pady=10)

        # Canvas for Drawing
        self.canvas_frame = tk.Frame(self.left_column, bd=2, relief="groove", bg="white")
        self.canvas_frame.pack(pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, bg="white", width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas.pack()

        # Draw the guide box
        self.guide_coords = (GUIDE_BOX_START_X, GUIDE_BOX_START_Y, GUIDE_BOX_END_X, GUIDE_BOX_END_Y)
        self.guide_rect_id = self.canvas.create_rectangle(
            *self.guide_coords,
            outline="green",
            dash=(5, 3),
            width=2,
            tags="guide_line"
        )
        self.guide_text_id = self.canvas.create_text(
            (GUIDE_BOX_START_X + GUIDE_BOX_END_X) / 2,
            GUIDE_BOX_START_Y - 15,
            text="Draw within this area",
            fill="gray",
            font=("Helvetica", 9),
            tags="guide_text"
        )

        # Initialize PIL Image
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # --- Controls Frame ---
        self.controls_frame = tk.Frame(self.left_column, bg=BACKGROUND_COLOR)
        self.controls_frame.pack(pady=10)

        # Brush Size Slider
        self.thickness_label = tk.Label(self.controls_frame, text="Brush Size:",
                                         font=("Helvetica", 10), bg=BACKGROUND_COLOR)
        self.thickness_label.pack(side=tk.LEFT, padx=5)

        self.thickness_slider = Scale(self.controls_frame, from_=10, to=30, orient=HORIZONTAL,
                                       command=self.update_thickness, showvalue=0,
                                       bg=BACKGROUND_COLOR)
        self.thickness_slider.set(self.drawing_line_width)
        self.thickness_slider.pack(side=tk.LEFT, padx=5)

        self.current_thickness_label = tk.Label(self.controls_frame,
                                                 text=str(self.drawing_line_width),
                                                 font=("Helvetica", 10), bg=BACKGROUND_COLOR)
        self.current_thickness_label.pack(side=tk.LEFT)

        # Button Frame - Now centered
        self.button_frame = tk.Frame(self.left_column, bg=BACKGROUND_COLOR)
        self.button_frame.pack(pady=10)

        # Inner frame for buttons to help with centering
        self.button_inner_frame = tk.Frame(self.button_frame, bg=BACKGROUND_COLOR)
        self.button_inner_frame.pack()

        # Predict Button
        self.predict_button = tk.Button(self.button_inner_frame, text="Predict Digit",
                                         command=self.predict_drawn_image,
                                         font=("Helvetica", 12), bg=BUTTON_COLOR, fg="white",
                                         padx=15, pady=5)
        self.predict_button.pack(side=tk.LEFT, padx=5)

        # Clear Button
        self.clear_button = tk.Button(self.button_inner_frame, text="Clear Canvas",
                                        command=self.clear_canvas,
                                        font=("Helvetica", 12), bg=CLEAR_COLOR, fg="white",
                                        padx=15, pady=5)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Save Image Button
        self.save_button = tk.Button(self.button_inner_frame, text="Save Image",
                                      command=self.save_canvas_image,
                                      font=("Helvetica", 12), bg=SAVE_COLOR, fg="white",
                                      padx=15, pady=5)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # About Button
        self.about_button = tk.Button(self.button_inner_frame, text="About",
                                       command=self.show_about,
                                       font=("Helvetica", 12), bg=ABOUT_COLOR, fg="white",
                                       padx=15, pady=5)
        self.about_button.pack(side=tk.LEFT, padx=5)

        # --- Right Column (Results) ---
        self.right_column = tk.Frame(self.main_frame, bg=BACKGROUND_COLOR)
        self.right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=10)

        # --- Top Right (Image and Prediction) ---
        self.top_right_frame = tk.Frame(self.right_column, bg=BACKGROUND_COLOR)
        self.top_right_frame.pack(fill=tk.X, pady=10)

        # Captured Image Display
        self.capture_frame = tk.Frame(self.top_right_frame, bg="white", bd=2, relief="groove")
        self.capture_frame.pack(side=tk.LEFT, padx=10)

        self.capture_label = tk.Label(self.capture_frame, text="Processed (28x28)",
                                        font=("Helvetica", 10), bg="white")
        self.capture_label.pack()

        self.capture_canvas = tk.Canvas(self.capture_frame, width=120, height=120, bg="white")
        self.capture_canvas.pack(pady=5)

        # Prediction Display
        self.prediction_frame = tk.Frame(self.top_right_frame, bg="white", bd=2, relief="groove")
        self.prediction_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        self.prediction_var = tk.StringVar()
        self.prediction_var.set("Draw a digit!")
        self.prediction_label = tk.Label(self.prediction_frame, textvariable=self.prediction_var,
                                         font=("Helvetica", 28, "bold"), bg="white",
                                         fg=BUTTON_COLOR, pady=20)
        self.prediction_label.pack()

        self.confidence_var = tk.StringVar()
        self.confidence_var.set("Confidence: N/A")
        self.confidence_label = tk.Label(self.prediction_frame, textvariable=self.confidence_var,
                                         font=("Helvetica", 14), bg="white")
        self.confidence_label.pack(pady=5)

        # Confidence Meter
        self.meter_frame = tk.Frame(self.prediction_frame, bg="white")
        self.meter_frame.pack(pady=5)

        self.meter_label = tk.Label(self.meter_frame, text="Confidence:",
                                     font=("Helvetica", 10), bg="white")
        self.meter_label.pack()

        self.meter_canvas = tk.Canvas(self.meter_frame, width=200, height=20, bg="white")
        self.meter_canvas.pack()
        self.meter = self.meter_canvas.create_rectangle(0, 0, 0, 20, fill=BUTTON_COLOR)

        # --- Bottom Right (Probability Chart) ---
        self.chart_frame = tk.Frame(self.right_column, bg=BACKGROUND_COLOR)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.figure = plt.Figure(figsize=(5, 4), dpi=100, facecolor=BACKGROUND_COLOR)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(BACKGROUND_COLOR)
        self.configure_chart_axes()
        self.bar_chart = FigureCanvasTkAgg(self.figure, self.chart_frame)
        self.bar_chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if self.model is None:
            self.predict_button.config(state=tk.DISABLED)
            self.prediction_var.set("Model Error!")
            self.confidence_var.set("Failed to load model")

    def show_about(self):
        """Show smooth, centered About dialog with properly sized button"""
        about_window = tk.Toplevel(self.master)
        about_window.withdraw()  # Hide until fully configured
        about_window.title("About Digit Recognizer")
        about_window.resizable(False, False)
        
        # Set icon if available for the Toplevel window
        try:
            about_window.iconbitmap(self.icon_file) # Use the same dynamically found icon
        except tk.TclError:
            pass
        
        # Main content container
        content_frame = tk.Frame(about_window, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Application info
        tk.Label(content_frame, text="Digit Recognizer", 
                 font=("Helvetica", 16, "bold")).pack(pady=5)
        
        tk.Label(content_frame, 
                 text="A simple MNIST digit recognition application\nusing TensorFlow and Tkinter",
                 font=("Helvetica", 10)).pack(pady=10)
        
        # Developer info
        tk.Label(content_frame, text="Developed by:", 
                 font=("Helvetica", 10, "bold")).pack(pady=5)
        
        dev_frame = tk.Frame(content_frame)
        dev_frame.pack()
        
        tk.Label(dev_frame, text="Amanuel Mihiret", font=("Helvetica", 10)).grid(row=0, column=0, sticky="w")
        
        # Email link
        email = tk.Label(dev_frame, text="zeaman48@gmail.com",
                         font=("Helvetica", 10), fg="blue", cursor="hand2")
        email.grid(row=1, column=0, sticky="w", pady=2)
        email.bind("<Button-1>", lambda e: webbrowser.open("mailto:zeaman48@gmail.com"))
        
        # GitHub link
        github = tk.Label(dev_frame, text="GitHub: github.com/zeaman",
                          font=("Helvetica", 10), fg="blue", cursor="hand2")
        github.grid(row=2, column=0, sticky="w", pady=2)
        github.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/zeaman"))
        
        # Close button container
        button_frame = tk.Frame(content_frame, pady=15)
        button_frame.pack(fill=tk.X)
        
        # Properly sized Close button
        close_button = tk.Button(
            button_frame, 
            text="Close", 
            command=about_window.destroy,
            width=12,
            padx=12,
            pady=4,
            font=("Helvetica", 10, "bold")
        )
        close_button.pack()
        
        # Calculate and set window size based on content
        about_window.update_idletasks()  # Calculate widget sizes
        
        # Get the requested width/height from the widgets
        width = content_frame.winfo_reqwidth() + 20  # Add padding
        height = content_frame.winfo_reqheight() + 20
        
        # Center the window relative to main window
        main_x = self.master.winfo_x()
        main_y = self.master.winfo_y()
        main_width = self.master.winfo_width()
        main_height = self.master.winfo_height()
        
        x = main_x + (main_width - width) // 2
        y = main_y + (main_height - height) // 2
        
        # Configure all window properties before showing
        about_window.geometry(f"{width}x{height}+{x}+{y}")
        about_window.transient(self.master)
        about_window.deiconify()  # Show the window
        about_window.grab_set()  # Make modal
        
        # Smooth focus transition
        about_window.after(10, lambda: about_window.focus_force())

    # --- Drawing Functions ---
    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y
        self.check_bounds(event.x, event.y)

    def draw_line(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                     width=self.drawing_line_width, fill="black",
                                     capstyle=tk.ROUND, smooth=tk.TRUE,
                                     tags="drawing_stroke")
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                            fill="black", width=self.drawing_line_width,
                            joint="round")
            self.last_x, self.last_y = event.x, event.y
            self.check_bounds(event.x, event.y)

    def stop_draw(self, event):
        self.last_x, self.last_y = None, None

    def update_thickness(self, value):
        self.drawing_line_width = int(value)
        self.current_thickness_label.config(text=str(self.drawing_line_width))

    def check_bounds(self, x, y):
        min_x, min_y, max_x, max_y = self.guide_coords
        half_brush = self.drawing_line_width / 2
        is_out = not (min_x + half_brush <= x <= max_x - half_brush and
                      min_y + half_brush <= y <= max_y - half_brush)
        if is_out and not self.drawing_out_of_bounds:
            self.drawing_out_of_bounds = True
            self.canvas.itemconfig(self.guide_rect_id, outline="red")

    def clear_canvas(self):
        self.canvas.delete("drawing_stroke")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_var.set("Draw a digit!")
        self.confidence_var.set("Confidence: N/A")
        self.meter_canvas.coords(self.meter, 0, 0, 0, 20)
        self.ax.clear()
        self.configure_chart_axes()
        self.bar_chart.draw()
        self.clear_captured_image()
        self.drawing_out_of_bounds = False
        self.canvas.itemconfig(self.guide_rect_id, outline="green")
        self.canvas.tag_raise("guide_text")

    def clear_captured_image(self):
        self.capture_canvas.delete("all")
        self.capture_canvas.create_rectangle(0, 0, 120, 120, fill="white", outline="gray")

    def save_canvas_image(self):
        try:
            # Use os.path.join for cross-platform compatibility
            output_dir = os.path.join(os.path.expanduser("~"), "saved_digits") # Save to user's home directory
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"digit_{timestamp}.png")
            self.image.save(filename)
            messagebox.showinfo("Image Saved", f"Digit image saved as:\n{filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image:\n{e}")

    def configure_chart_axes(self):
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(-0.5, 9.5)
        self.ax.set_xticks(range(10))
        self.ax.set_xlabel('Digits (0-9)', fontsize=9)
        self.ax.set_ylabel('Probability', fontsize=9)
        self.ax.set_title('Digit Probability Distribution', fontsize=11, pad=10)
        self.ax.grid(True, linestyle='--', alpha=0.6)

    def preprocess_drawn_image(self):
        # Check for empty canvas
        if np.all(np.array(self.image) == 255):
            messagebox.showwarning("Empty Canvas", "Please draw a digit first")
            return None

        crop_region = (GUIDE_BOX_START_X, GUIDE_BOX_START_Y, GUIDE_BOX_END_X, GUIDE_BOX_END_Y)
        img_cropped = self.image.crop(crop_region)
        img_resized = img_cropped.resize((PREDICTION_IMAGE_SIZE, PREDICTION_IMAGE_SIZE),
                                         Image.Resampling.LANCZOS)
        img_array = 255 - np.array(img_resized)
        self.display_processed_image(img_array)
        img_normalized = img_array.astype("float32") / 255.0

        return img_normalized.reshape(1, PREDICTION_IMAGE_SIZE, PREDICTION_IMAGE_SIZE, 1)

    def display_processed_image(self, img_array):
        img = Image.fromarray(img_array)
        img_display = img.resize((100, 100), Image.Resampling.NEAREST)
        self.capture_img = ImageTk.PhotoImage(image=img_display)
        self.capture_canvas.delete("all")
        self.capture_canvas.create_image(60, 60, image=self.capture_img)

    def predict_drawn_image(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Model is not loaded. Cannot make predictions.")
            return

        if self.drawing_out_of_bounds:
            if not messagebox.askyesno(
                "Drawing Out of Bounds",
                "Your drawing extends beyond the guide area.\nProceed with prediction anyway?"
            ):
                return

        processed_image = self.preprocess_drawn_image()
        if processed_image is None:
            return

        try:
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_digit = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            # Update UI
            if confidence < CONFIDENCE_THRESHOLD:
                self.prediction_var.set("Unknown")
                self.prediction_label.config(fg="red")
            else:
                self.prediction_var.set(str(predicted_digit))
                self.prediction_label.config(fg=BUTTON_COLOR)

            self.confidence_var.set(f"Confidence: {confidence * 100:.1f}%")
            self.meter_canvas.coords(self.meter, 0, 0, 200 * confidence, 20)

            # Update chart
            self.ax.clear()
            bars = self.ax.bar(range(10), predictions[0], color='#4CAF50')
            bars[predicted_digit].set_color('#2196F3')
            for i, v in enumerate(predictions[0]):
                self.ax.text(i, v + 0.02, f"{v:.2f}", color='black', ha='center', fontsize=8)
            self.configure_chart_axes()
            self.bar_chart.draw()

        except Exception as e:
            error_msg = f"Prediction Error:\n\n{str(e)}"
            if self.model is not None and hasattr(self.model, 'input_shape'):
                error_msg += f"\n\nExpected shape: {self.model.input_shape}"
                if 'processed_image' in locals():
                    error_msg += f"\nActual shape: {processed_image.shape}"
            messagebox.showerror("Prediction Error", error_msg)
            self.prediction_var.set("Error!")
            self.confidence_var.set("See error message")
            self.prediction_label.config(fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitPredictorApp(root)
    root.mainloop()