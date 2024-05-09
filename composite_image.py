from facer import facer
import matplotlib.pyplot as plt
import os

image_folder = 'images' # i/o folder for generated images
my_project_path = os.path.dirname(os.path.abspath(__file__))
my_faces_path = my_project_path + os.path.sep + image_folder + os.path.sep
images = facer.load_images(my_faces_path)

landmarks, faces = facer.detect_face_landmarks(images)

facer.create_average_face(faces, landmarks, save_image=True)

# facer.create_animated_gif(my_faces_path)
