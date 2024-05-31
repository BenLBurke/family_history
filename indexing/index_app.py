import gradio as gr
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from PIL import Image
import requests
from io import BytesIO

# Load the dataset from Hugging Face Hub
dataset = load_dataset("straka/distinct_words")

# Initialize an index to track the current image
index = 0

# Function to get the current image and name
def get_image_and_name(index):
    image_info = dataset["train"][index]
    image = image_info["image"]
    image_name = image_info["image_name"]
    return image, image_name

# Function to update the image based on the index and save the typed word
def update_image(image_index, typed_word, typed_words, image_names):
    image, image_name = get_image_and_name(image_index)
    
    # Save the typed word and image name
    typed_words.append(typed_word)
    image_names.append(image_name)
    
    return image, image_name, "", typed_word, typed_words, image_names

# Function to go to the next image
def next_image(current_index, typed_word, typed_words, image_names):
    new_index = current_index + 1
    if new_index >= len(dataset["train"]):
        new_index = 0
    return update_image(new_index, typed_word, typed_words, image_names)

# Function to go to the previous image
def previous_image(current_index, typed_word, typed_words, image_names):
    new_index = current_index - 1
    if new_index < 0:
        new_index = len(dataset["train"]) - 1
    return update_image(new_index, typed_word, typed_words, image_names)

# Create a Gradio interface
with gr.Blocks() as demo:
    image_display = gr.Image(type="pil")
    image_name_display = gr.Textbox(label="Image Name")
    word_input = gr.Textbox(label="Type the word you see in the image")
    current_index = gr.Number(value=0, label="Image Index", interactive=False)
    typed_words = gr.State([])
    image_names = gr.State([])

    next_button = gr.Button("Next")
    prev_button = gr.Button("Previous")

    prev_button.click(previous_image, inputs=[current_index, word_input, typed_words, image_names], outputs=[image_display, image_name_display, word_input, typed_words, image_names, current_index])
    next_button.click(next_image, inputs=[current_index, word_input, typed_words, image_names], outputs=[image_display, image_name_display, word_input, typed_words, image_names, current_index])

    # Initialize the first image
    demo.load(update_image, inputs=[current_index, word_input, typed_words, image_names], outputs=[image_display, image_name_display, word_input, typed_words, image_names, current_index])

# Launch the app
demo.launch()
