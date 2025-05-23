"""
Gender Classification

Description:
Script that assigns binary gender labels (man or woman) to images
using a gender classification algorithm.
The dashboard (main.py) uses this script to determine the gender of the user's
generated images, which in turn allows users to see the gender distribution
of their generated variations.

Features:
- Gender classification and labeling

Models used:
- OpenAI CLIP ViT-B/32: Gender classification

Usage examples:
classify_gender('img-1.jpg')                 - Single image path
classify_gender(['img-1.jpg', 'img-2.jpg'])  - List of image paths
classify_gender('images')                    - Folder path

Author: Mike Ebbe
Institution: HU University of Applied Sciences
Version: 1.0
"""

""" Package imports
"""
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os


""" Global variables:
    These values may be adjusted to preference.
"""
# Load model via Huggingface
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load model locally
# model = CLIPModel.from_pretrained("./models/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("./models/clip-vit-base-patch32")


""" This function uses OpenAI's CLIP ViT-B/32 model to perform gender classification
    on one or multiple input images.
    :param input: The path of the image or image folder, or a list of image paths.
    :return predictions: A list of dictionaries containing the gender predictions,
    along with a confidence score.
"""
def classify_gender(input):
    labels = ['man', 'woman']

    if isinstance(input, list):
        # Input is a list of images
        images = []
        image_names = []
        for image_path in input:
            if os.path.isfile(image_path):
                images.append(Image.open(os.path.join(image_path)))
                image_names.append(os.path.basename(image_path))
    elif os.path.isfile(input):
        # Input is one image
        images = Image.open(os.path.join(input))
        image_names = [os.path.basename(input)]
    elif os.path.isdir(input):
        # Input is a folder of images
        jpg_images = [file for file in os.listdir(input) if file.endswith(".jpg")]
        images = [Image.open(os.path.join(input, image)) for image in jpg_images]
        image_names = jpg_images
    else:
        # Error
        with open('logs/error_log.txt', 'a') as file:
                file.write(f'Image or folder {input} not found')
    
    inputs = processor(
        text=labels,
        images=images,
        return_tensors="pt",
        padding=True
    )

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # This is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    predictions = []
    for i, prediction in enumerate(probs):
        pred_value, pred_index = prediction.max(0) # Select the label with the highest confidence score
        predictions.append({
            'name': image_names[i], # Name of image
            'label': labels[pred_index.item()], # Gender label
            'confidence': pred_value.item() # Confidence score
        })

    del input, inputs, outputs, logits_per_image, probs

    return predictions
