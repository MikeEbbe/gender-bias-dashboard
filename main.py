from datetime import datetime
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms
import glob
import gradio as gr
import os
import shutil

device = 'cpu' # TODO: change to GPU when running on research drive
grid_size = 10 # amount of images to generate
columns = grid_size / 2
rows = grid_size / columns
image_folder = 'demo_images'
image_variation_folder = 'image_variations'
iteration = 0
log_file = "logs/gradio_log.txt"

""" This function returns a list of syntetic images
    The images have to be added manually to the demo_images 
    folder.
"""
def get_synthetic_images():
    images = []
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
        image_path = os.path.join(image_folder, evt.value['image']['orig_name'])
    elif iteration > 1:
        image_path = os.path.join(image_variation_folder, evt.value['image']['orig_name'])
    image_variations = generate_variation(image_path, dummy=True)
    print(image_variations)
    return image_variations

""" This function generates a new variation of the input image
:param image_path: The path of the selected image
"""
def generate_variation(image_path, dummy):
    if dummy:
        """Simulate generating an image. This is used for testing purposes,
           for when the user doesn't have the hardware capable of running
           Stable Diffusion efficiently.
        """
        image_variations = []
        for i in range(grid_size):
            file_path = os.path.join(image_variation_folder, datetime.now().strftime("%Y%m%d%H%M%S")) + str(i) + ".jpg"
            shutil.copyfile(image_path, file_path)
            image_variations.append(file_path)
        return image_variations
    else:
        sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            revision="v2.0",
        )
        sd_pipe = sd_pipe.to(device)
        input = pre_process_image(image_path)
        output = sd_pipe(input, guidance_scale=7, num_images_per_prompt=grid_size)
        image_variations = []
        for i, output_image in enumerate(output["images"]):
            file_path = os.path.join(image_variation_folder, datetime.now().strftime("%Y%m%d%H%M%S")) + str(i) + ".jpg"
            output_image.save(file_path)
            image_variations.append(file_path)
        return image_variations
    

""" This function applies preprocessing to the image using
    the PyTorch package.
"""
def pre_process_image(image_path):
    print(image_path)
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
            name = gr.Textbox(label="Prompt")
            gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery",
                columns=columns, rows=rows, object_fit="contain", height="auto", min_width=200, allow_preview=False
            )
            gallery.select(select_image, None, outputs=gallery)
            generate_btn = gr.Button("Generate new set of images")
            generate_btn.click(get_synthetic_images, None, outputs=gallery)
        text1 = gr.Textbox(label="Metrics")

if __name__ == "__main__":
    demo.launch()
