from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os


def classify_gender(input):
    # Via Huggingface
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Locally
    model = CLIPModel.from_pretrained("./model/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("./model/clip-vit-base-patch32")

    labels = ['man', 'woman']

    if isinstance(input, list):
        # Input is a list of images
        # print('input is list of images')
        images = []
        image_names = []
        for image_path in input:
            if os.path.isfile(image_path):
                images.append(Image.open(os.path.join(image_path)))
                image_names.append(os.path.basename(image_path))
    elif os.path.isfile(input):
        # Input is one image
        # print('input is image')
        images = Image.open(os.path.join(input))
        image_names = [os.path.basename(input)]
        # print(images)
    elif os.path.isdir(input):
        # Input is a folder of images
        # print('input is folder')
        jpg_images = [file for file in os.listdir(input) if file.endswith(".jpg")]
        images = [Image.open(os.path.join(input, image)) for image in jpg_images]
        image_names = jpg_images
        # print(images)
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

    # print(probs)
    predictions = []
    for i, prediction in enumerate(probs):
        pred_value, pred_index = prediction.max(0) # Select the label with the highest confidence score
        # print(f'Image: {image_names[i]} - Label: {labels[pred_index.item()]} - Confidence: {pred_value.item()}')
        predictions.append({
            'name': image_names[i], # Name of image
            'label': labels[pred_index.item()], # Gender label
            'confidence': pred_value.item() # Confidence score
        })

    return predictions

# classify_gender(['images/demo-1.jpg', 'images/demo-2.jpg'])
# classify_gender('images')
# classify_gender('images/demo-3.jpg')
