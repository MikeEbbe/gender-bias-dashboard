"""
Package imports
"""
from datetime import datetime
from diffusers import StableDiffusionPipeline, StableDiffusionImageVariationPipeline, DPMSolverMultistepScheduler
from PIL import Image
from torchvision import transforms
import gradio as gr
import math
import os
import shutil
import torch


"""
Global variables:
These values can be adjusted to preference.
"""
device = 'cpu' # TODO: change to GPU when running on SURF
dummy = True # used for testing purposes. When enabled, instead of generating images, dummy images are used.
grid_size = 10 # amount of images to generate, and show in the grid
image_folder = 'images' # i/o folder for generated images
image_variation_folder = 'image_variations' # i/o folder for generated variations
log_file = "logs/error_log.txt" # log file for errors


"""
Intialization variables:
These values, while global, should not be changed. 
"""
columns = math.floor(grid_size / 2)
rows = math.floor(grid_size / columns)
iteration = 0


""" This function handles the click action on the generate button.
:param prompt: The text value of the prompt textbox.
"""
def click_generate_button(prompt):
    global iteration
    iteration = 0 # reset variation iteration
    if dummy:
        images = get_synthetic_images()
        return images
    else:
        # sd
        images = generate_images(prompt)
        return images


""" This function generates an n amount of images using Stable Diffusion.
    The images are saved to a folder, and the paths to these saved images are
    returned in a list.
    Stable Diffusion model: https://huggingface.co/stabilityai/stable-diffusion-2-1
:param prompt: The text prompt to generate images for.
"""
def generate_images(prompt):
    # print(prompt)
    # Load the Stable Diffusion model
    sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe = sd_pipe.to(device)

    images = []

    # Generate images for the given prompt
    for i, image in enumerate(sd_pipe(prompt, num_images_per_prompt=grid_size).images):
        file_path = os.path.join(image_folder, datetime.now().strftime("%Y%m%d%H%M%S")) + str(i) + ".jpg"
        image.save(file_path)
        images.append(file_path)
    return images


""" This function returns a list of 10 syntetic images
    It's possible to load more images by manually adding
    them to the images folder.
"""
def get_synthetic_images():
    images = []

    # Load the synthetic images
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        images.append((image_path, 'label'))
    return images


""" This function handles the click action on an image in the gallery
:param evt: An object containing the event data 
"""
def select_image(evt: gr.SelectData):
    global iteration
    iteration += 1

    if iteration == 1:
        # Use the images folder
        image_path = os.path.join(image_folder, evt.value['image']['orig_name'])
    elif iteration > 1:
        # Use the image variation folder
        image_path = os.path.join(image_variation_folder, evt.value['image']['orig_name'])

    image_variations = generate_variations(image_path)
    # print(image_variations)
    return image_variations


""" This function generates a new variation of the input image
:param image_path: The path of the selected image
"""
def generate_variations(image_path):
    if dummy:
        image_variations = []

        # Duplicate the selected image n times
        for i in range(grid_size):
            file_path = os.path.join(image_variation_folder, datetime.now().strftime("%Y%m%d%H%M%S")) + str(i) + ".jpg"
            shutil.copyfile(image_path, file_path)
            image_variations.append(file_path)
        return image_variations
    else:
        # Load the Stable Diffusion Variation model
        sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            revision="v2.0",
        )
        sd_pipe = sd_pipe.to(device)

        input = pre_process_image(image_path)

        # Generate n images
        output = sd_pipe(input, guidance_scale=7, num_images_per_prompt=grid_size)

        image_variations = []

        # Save the images
        for i, output_image in enumerate(output["images"]):
            file_path = os.path.join(image_variation_folder, datetime.now().strftime("%Y%m%d%H%M%S")) + str(i) + ".jpg"
            output_image.save(file_path)
            image_variations.append(file_path)
        return image_variations
    

""" This function applies pre-processing to the image using
    the PyTorch package.
:param image_path: The location of the image to pre-process
"""
def pre_process_image(image_path):
    # print(image_path)
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
    return input


""" This function removes all files form the image variation folder
    TODO: integrate this into the code, and write a similar function 
    for the images folder.
"""
def clear_image_variation_folder():
    for filename in os.listdir(image_variation_folder):
        # print(filename)
        file_path = os.path.join(image_variation_folder, filename)
        try:
            os.unlink(file_path)
        except Exception as e:
            with open('logs/gradio_log.txt', 'a') as file:
                file.write('Failed to delete %s. Reason: %s' % (file_path, e))


""" This block of code makes up the interface
"""
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=1000):
            prompt_input = gr.Textbox(label="Prompt")
            gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery",
                columns=columns, rows=rows, object_fit="contain", height="auto", min_width=200, allow_preview=False
            )
            gallery.select(select_image, None, outputs=gallery)
            generate_btn = gr.Button("Generate new set of images")
            generate_btn.click(click_generate_button, inputs=prompt_input, outputs=gallery)
        text1 = gr.Textbox(label="Metrics")


""" Initialize
"""
if __name__ == "__main__":
    demo.launch(favicon_path='favicon.ico')
