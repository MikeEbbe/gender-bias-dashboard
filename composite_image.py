"""
Image Compositing

Description:
Script that combines separate images of people's faces into one average image.
The dashboard does not use this script, as it was not implemented into the final design.
Future studies may further develop using composite images as a way of presenting bias
to users and measuring subsequent user behaviour.

Features:
- Composite image generation

Packages used:
- Facer: https://github.com/johnwmillr/Facer

Usage example:
generate_composite('images')  - Folder path

Author: Mike Ebbe
Institution: HU University of Applied Sciences
Version: 1.0
"""

""" Package imports
"""
from facer import facer
import matplotlib.pyplot as plt
import os


""" This function uses OpenCV to create a composite image from multiple images.
    :param input: I/O folder that contains faces, and is used for storing the composite image.
"""
def generate_composite(input):
    my_project_path = os.path.dirname(os.path.abspath(__file__))
    my_faces_path = my_project_path + os.path.sep + input + os.path.sep
    images = facer.load_images(my_faces_path)

    landmarks, faces = facer.detect_face_landmarks(images)

    facer.create_average_face(faces, landmarks, save_image=True)

    # facer.create_animated_gif(my_faces_path) # Create animated GIF
