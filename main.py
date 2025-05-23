"""
Interactive Image Generation Dashboard

Description:
Web application that allows users to iteratively generate and refine AI-generated images,
while providing real-time feedback on gender representation.
Users start with a prompt, select from 8 generated images, and create variations to
explore how their choices affect gender diversity in AI-generated content.

Features:
- Image generation using the Stable Diffusion model
- Real-time gender classification and labeling (gender_classifier.py)
- Visual feedback through stacked bar charts showing the gender ratio
- Iterative refinement process with user-guided selection
- Prompt and image storage for research purposes

Models used:
- Stable Diffusion 2.1: Base image generation
- Stable Diffusion 1.4 (finetuned): Image variations
- OpenAI CLIP ViT-B/32: Gender classification

How to Run:
    1. Install the dependencies in requirements.txt
    2. Run gradio main.py
    3. Open the provided URL in your browser to access the dashboard

Author: Mike Ebbe
Institution: HU University of Applied Sciences
Version: 1.0
"""

""" Package imports
"""
from datetime import datetime
from diffusers import StableDiffusionPipeline, StableDiffusionImageVariationPipeline, DPMSolverMultistepScheduler
from PIL import Image
from re import sub
from torchvision import transforms
import glob
import gradio as gr
import math
import os
import plotly.graph_objects as go
import random
import shutil
import torch
import uuid

""" File imports
"""
from gender_classifier import classify_gender
# from composite_image import generate_composite


""" Global variables:
    These values may be adjusted to preference.
"""
device = 0 # change to cpu or cuda depending on environment
dummy = False # used for testing purposes. When enabled, instead of generating images, dummy images are used.
grid_size = 8 # amount of images to generate, and show in the grid

""" Initialization variables:
    These values should not be changed. 
"""
dummy_image_folder = "results/dummy_user/images/a_photograph_of_a_ceo" # i/o folder for generated images
dummy_image_variation_folder = "results/dummy_user/images/a_photograph_of_a_ceo/variations" # i/o folder for generated variations
columns = max(1, math.floor(grid_size / 2)) # Amount of columns of the image gallery (min. 1)
rows = math.floor(grid_size / columns) # Amount of rows of the image gallery
bar_chart = gr.Plot(label="Plot", elem_id="bar_chart", show_label=False) # metrics

# Load the Stable Diffusion model
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
sd_pipe = sd_pipe.to(device)

# Load the Stable Diffusion Variation model
sdv_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    revision="v2.0",
    torch_dtype=torch.float16,
)
sdv_pipe = sdv_pipe.to(device)


""" This function handles the click action on the generate button.
    :param prompt: The text value of the prompt textbox.
    :return images: The list of image/label tuples.
    :return images_with_gender_labels: The list of image/gender tuples.
    :return gender_chart: Bar chart with gender distribution.
"""
def click_generate_button(prompt, user_id, iteration, all_images_with_labels, current_prompt, dashboard):
    current_prompt = prompt
    iteration = 0 # reset variation iteration
    if dummy:
        images = get_synthetic_images() # Images and labels
    else:
        # generate images with Stable Diffusion
        images = generate_images(prompt, user_id)
    
    if dashboard == "metrics":
        images_with_gender_labels = add_labels(images)
        all_images_with_labels.extend(images_with_gender_labels)
        gender_chart = generate_bar_chart(all_images_with_labels)
        return images_with_gender_labels, gender_chart, iteration, all_images_with_labels, current_prompt, dashboard
    
    return images, None, iteration, all_images_with_labels, current_prompt, dashboard


""" This function converts a string to lower snake case.
    Snippet written by w3resource (https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-97.php).
    :param s: String to convert.
    :return: The converted string in lower snake_case format.
"""
def snake_case(s):
    # Replace hyphens with spaces, then apply regular expression substitutions for title case conversion
    # and add an underscore between words, finally convert the result to lowercase
    return '_'.join(
        sub('([A-Z][a-z]+)', r' \1',
        sub('([A-Z]+)', r' \1',
        s.replace('-', ' '))).split()).lower()


""" This function generates an n amount of images using Stable Diffusion.
    The images are saved to a folder, and the paths to these saved images are
    returned in a list.
    :param prompt: The text prompt to generate images for.
    :return images: The list of image/label tuples.
"""
def generate_images(prompt, user_id):
    images = []

    # Generate one male and one female image to ensure users can choose between both genders.
    # This addresses potential model biases that might otherwise result in images of only one gender,
    # limiting the user's choices.
    male_image = sd_pipe(prompt + " male", num_images_per_prompt=1).images[0]
    female_image = sd_pipe(prompt + " female", num_images_per_prompt=1).images[0]
    unspecified_images = sd_pipe(prompt, num_images_per_prompt=grid_size - 2).images
    all_images = [male_image, female_image] + unspecified_images
    random.shuffle(all_images) # randomize order
    
    dir_path = os.path.join('results', str(user_id), 'images', snake_case(prompt))
    os.makedirs(dir_path, exist_ok=True) # create image directory for the user

    # Save the generated images
    for i, image in enumerate(all_images):
        file_path = os.path.join(dir_path, f'{str(i)}.jpg')
        image.save(file_path)
        images.append((file_path, "label"))  

    del male_image, female_image, unspecified_images, all_images
    torch.cuda.empty_cache()

    return images


""" This function adds gender labels to each image.
    It uses a CLIP model to predict the gender.
    :param images: The list of image/label tuples.
    :return images_with_gender_labels: The list of image/gender tuples.
"""
def add_labels(images):
    image_list = [t[0] for t in images] # Only the image paths
    gender_metrics = classify_gender(image_list) # Predict the gender

    # Update the gender labels
    images_with_gender_labels = [(image[0], metric["label"]) for image, metric in zip(images, gender_metrics)]
    return images_with_gender_labels


""" This function creates a bar chart using the Plotly package.
    It creates a stacked bar chart to visualize the distribution
    between the "man" and "woman" variables.
    :param images_with_gender_labels: The list of image/gender tuples.
    :return plot: Bar chart with gender distribution.
"""
def generate_bar_chart(images_with_gender_labels):
    # Get percentage of gender
    men_count = sum(label == "man"for _, label in images_with_gender_labels)
    women_count = sum(label == "woman" for _, label in images_with_gender_labels)
    total_people = len(images_with_gender_labels)
    men_percentage = round((men_count / total_people) * 100, 0)
    women_percentage = round((women_count / total_people) * 100, 0)

    x = ["Gender distribution"]
 
    # Create bar chart plot
    plot = go.Figure(data=[
        go.Bar(
            name = "Man",
            x = x,
            y = [men_percentage],
            width=0.8,
        ),
        go.Bar(
            name = "Woman",
            x = x,
            y = [women_percentage],
            width=0.8,
        )   
    ])
    
    plot.update_layout(
        autosize=False,
        barmode="stack",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Gender distribution",
        width=300,
        xaxis={'visible': False, 'showticklabels': False}
    )
                    
    return plot


""" This function returns a list of 10 synthetic images.
    It is possible to load more images by manually adding
    them to the dummy images folder.
    :return images: List of image/label tuples.
"""
def get_synthetic_images():
    images = []

    # Load the synthetic images
    for file in os.listdir(dummy_image_folder):
        if file.endswith(".jpg"):
            image_path = os.path.join(dummy_image_folder, file)
            images.append((image_path, "label"))
    return images


""" This function handles the click action on an image in the gallery.
    :param evt: An object containing the event data.
    :return image_variations: The list of variation/gender tuples.
"""
def select_image(evt: gr.SelectData, user_id, iteration, current_prompt, dashboard, all_images_with_labels):
    iteration += 1

    if iteration == 1:
        # Use the images folder
        if dummy:
            image_path = os.path.join(dummy_image_folder, evt.value["image"]["orig_name"])
        else:
            image_path = os.path.join('results', str(user_id), 'images', snake_case(current_prompt), evt.value["image"]["orig_name"])
    elif iteration > 1:
        # Use the image variation folder
        if dummy:
            image_path = os.path.join(dummy_image_variation_folder, evt.value["image"]["orig_name"])
        else:
            image_path = os.path.join('results', str(user_id), 'images', snake_case(current_prompt), 'variations', f'iteration_{iteration-1}', evt.value["image"]["orig_name"])
    image_variations = generate_variations(image_path, user_id, iteration, current_prompt)

    if dashboard == "metrics":
        images_variations_with_gender_labels = add_labels(image_variations)
        all_images_with_labels.extend(images_variations_with_gender_labels)
        gender_chart = generate_bar_chart(all_images_with_labels)
        return images_variations_with_gender_labels, gender_chart, iteration, all_images_with_labels, current_prompt, dashboard

    return image_variations, None, iteration, all_images_with_labels, current_prompt, dashboard


""" This function generates a new variation of the input image.
    :param image_path: The path of the selected image.
    :return image_variations: The list of variation/label tuples.
"""
def generate_variations(image_path, user_id, iteration, current_prompt):
    if dummy:
        paths_list = glob.glob(dummy_image_variation_folder + '/**/*.jpg', recursive=True)
        files_list = [path for path in paths_list if os.path.isfile(path)]

        random.shuffle(files_list)
        image_variations = [(path, 'label') for path in files_list[:10]]

        return image_variations
    else:
        # copy image to variation folder for ease of access
        dir_path = os.path.join('results', str(user_id), 'images', snake_case(current_prompt), 'variations', f'iteration_{iteration}') # variation directory path
        os.makedirs(dir_path, exist_ok=True) # create variation directory for the user
        shutil.copy(image_path, os.path.join(dir_path, 'base.png'))
        input = pre_process_image(image_path)

        # Generate n images
        output = sdv_pipe(input, num_inference_steps=20, guidance_scale=2, num_images_per_prompt=grid_size)

        image_variations = []

        # Save the images
        for i, output_image in enumerate(output["images"]):
            file_path = os.path.join(dir_path, f'{i}.png')
            
            output_image.save(file_path)
            image_variations.append((file_path, "label"))

        del input, output
        torch.cuda.empty_cache()

        return image_variations
    

""" This function applies pre-processing to an image using
    the PyTorch package.
    :param image_path: The location of the image to pre-process.
    :return input: Pre-processed image.
"""
def pre_process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(), # convert the image to a tensor
        transforms.Resize( # downscale the image
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        ),
        transforms.Normalize( # normalize the image
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]
        ),
    ])
    input = transform(image).to(device).unsqueeze(0) # apply transformation to image
    del image

    return input


""" This function checks the URL parameter to decide if the user
    should be shown the control or experimental dashboard.
    :param request: The incoming request object.
    :param user_id: Optional unique user identifier, will be generated if None.
    :return user_id: The updated user_id.
    :return dashbord: The dashboard type to show (metrics or regular).
"""
def check_url_param(request: gr.Request, user_id, dashboard):
    if user_id is None:
        user_id = uuid.uuid4() # assign ID
    query_params = dict(request.query_params)
    if 'a' in query_params and query_params['a'] == '1':
        # experimental dashboard
        dashboard = 'metrics'
        user_id = str(user_id) + '_experimental'
    else:
        # control dashboard
        dashboard = 'regular'
        user_id = str(user_id) + '_control'
    return user_id, dashboard


""" This block of code makes up the interface.
"""
with gr.Blocks(title="Image generation dashboard", css=".js-plotly-plot {display: flex; justify-content: center}") as demo:
    user_id_var = gr.State(None)
    iteration_var = gr.State(0)
    current_prompt_var = gr.State("")
    all_images_with_labels_var = gr.State([])
    dashboard_var = gr.State("regular")
    
    j = gr.JSON(visible=False)
    demo.load(check_url_param, inputs=[user_id_var, dashboard_var], outputs=[user_id_var, dashboard_var])
    with gr.Row():
        with gr.Column(scale=1, min_width=1000):
            prompt_input = gr.Textbox(label="Prompt")
            gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery",
                columns=columns, rows=rows, object_fit="contain", height="auto", min_width=200,
                allow_preview=False, interactive=False
            )
            gallery.select(select_image, 
                inputs=[user_id_var, iteration_var, current_prompt_var, dashboard_var, all_images_with_labels_var],
                outputs=[gallery, bar_chart, iteration_var, all_images_with_labels_var, current_prompt_var, dashboard_var]
            )
            generate_btn = gr.Button("Generate new set of images")
            generate_btn.click(
                click_generate_button, 
                inputs=[prompt_input, user_id_var, iteration_var, all_images_with_labels_var, current_prompt_var, dashboard_var],
                outputs=[gallery, bar_chart, iteration_var, all_images_with_labels_var, current_prompt_var, dashboard_var]
            )
        with gr.Column(scale=1):
            bar_chart.render()


""" Initialize
"""
if __name__ == "__main__":
    demo.launch(favicon_path="favicon.ico", share=True)
