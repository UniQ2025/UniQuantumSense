import unittest
import numpy as np
from unittest.mock import patch
import tkinter as tk
from UniQuantumSense_face_detection import (
    hash_image,
    load_image,
    processed_hashes,
    display_image,
)

class TestFaceDetectionApp(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window

    def tearDown(self):
        self.root.destroy()

    def test_hash_image(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        image_hash = hash_image(img)
        self.assertIsInstance(image_hash, str)

    @patch('UniQuantumSense_face_detection.simpledialog.askstring')
    @patch('UniQuantumSense_face_detection.filedialog.askopenfilename')
    @patch('UniQuantumSense_face_detection.cv2.imread')
    @patch('UniQuantumSense_face_detection.messagebox.showinfo')
    def test_load_image_faces(self, mock_showinfo, mock_imread, mock_askopenfilename, mock_askstring):
        # Simulate user input for "Load Image" and then "Faces"
        mock_askstring.return_value = 'Faces'
        mock_askopenfilename.return_value = 'test_face_image.jpg'
        mock_imread.return_value = np.zeros((224, 224, 3), dtype=np.uint8)  # Dummy image

        load_image()  # Call the function

        # Check if the expected message is shown
        mock_showinfo.assert_called_with("Info", "Face detected")

    @patch('UniQuantumSense_face_detection.simpledialog.askstring')
    @patch('UniQuantumSense_face_detection.filedialog.askopenfilename')
    @patch('UniQuantumSense_face_detection.cv2.imread')
    @patch('UniQuantumSense_face_detection.messagebox.showinfo')
    def test_load_image_non_faces(self, mock_showinfo, mock_imread, mock_askopenfilename, mock_askstring):
        # Simulate user input for "Load Image" and then "Non_Faces"
        mock_askstring.return_value = 'Non_Faces'
        mock_askopenfilename.return_value = 'test_non_face_image.jpg'
        mock_imread.return_value = np.zeros((224, 224, 3), dtype=np.uint8)  # Dummy image

        load_image()  # Call the function

        # Check if the expected message is shown
        mock_showinfo.assert_called_with("Info", "No face detected")

    def test_display_image(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)  # Dummy image
        try:
            display_image(img)  # Call display_image with just the image
            self.assertTrue(hasattr(display_image, 'panel'))  # Check if the panel was created
        except Exception as e:
            self.fail(f"display_image raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()