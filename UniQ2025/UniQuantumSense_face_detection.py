import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import simpledialog
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import hashlib

# Load the trained model
model = tf.keras.models.load_model('facial_detection_model.keras')

# Set to store hashes of processed images
processed_hashes = set()

# Global variable for the Tkinter window
window = None

# Function to compute the hash of an image
def hash_image(image):
    image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
    return hashlib.sha256(image_bytes).hexdigest()

# Load and process the image
def load_image():
    # Ask the user whether they want to load a Face or Non-Face image
    choice = simpledialog.askstring("Select Type", "Enter 'Faces' or 'Non_Faces':")
    if choice is None:
        return  # User canceled the dialog

    choice = choice.strip().lower()
    if choice not in ['faces', 'non_faces']:
        messagebox.showerror("Error", "Invalid selection. Please enter 'Faces' or 'Non_Faces'.")
        return

    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Could not load image.")
        return

    image_hash = hash_image(img)
    if image_hash in processed_hashes:
        messagebox.showinfo("Result", "This image has already been processed.")
        return
    else:
        processed_hashes.add(image_hash)

    resized_img = cv2.resize(img, (224, 224))
    normalized_img = resized_img / 255.0
    input_img = np.expand_dims(normalized_img, axis=0)

    # Predict if there is a face in the image
    prediction = model.predict(input_img)
    is_face_detected = prediction[0][0] > 0.5

    # Determine the expected result based on user selection
    if choice == 'faces':  # User selected "Faces"
        if is_face_detected:
            messagebox.showinfo("Result", "Face detected!")
        else:
            messagebox.showinfo("Result", "No face detected.")
    else:  # User selected "Non_Faces"
        if is_face_detected:
            messagebox.showinfo("Result", "Unexpected face detected.")
        else:
            messagebox.showinfo("Result", "No face detected as expected.")

    display_image(img)

# Function to display the image in the Tkinter window
def display_image(img):
    global window

    window = tk.Toplevel()
    window.title("Image Display")

    if hasattr(display_image, 'panel') and display_image.panel is not None:
        try:
            display_image.panel.destroy()
        except Exception as e:
            print(f"Error destroying previous image panel: {e}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_resized = img_pil.resize((400, 300), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_resized)
    display_image.panel = tk.Label(window, image=img_tk)
    display_image.panel.image = img_tk
    display_image.panel.grid(row=1, column=0, columnspan=2)

    window.deiconify()
    window.update_idletasks()

# Create the main window
window = tk.Tk()
window.title("Face Detection Application")

# Create a button to load an image
load_button = tk.Button(window, text="Load Image", command=load_image)
load_button.grid(row=0, column=0, padx=10, pady=10)

# Start the Tkinter main loop
window.mainloop()