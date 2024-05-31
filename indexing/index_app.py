import gradio as gr
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from PIL import Image
import requests
from io import BytesIO

# Load the dataset from Hugging Face Hub
word_dataset = load_dataset("straka/distinct_words")
source_dataset = load_dataset("straka/source_pages")

# Initialize an index to track the current image
index = 0

# Function to get the current image and name
def get_image_and_name(dataset:Dataset, index:int):
    image_info = word_dataset["train"][index]
    image = image_info["image"]
    image_name = image_info["image_name"]
    return image, image_name

# Function to update the image based on the index and save the typed word
def update_image(image_index, typed_word, typed_words, image_names):
    print(f"Updated Image index: {image_index}")
    word_image, image_name = get_image_and_name(word_dataset, image_index)

    # source_image_name = image_name.split('_')[0] + '_' + image_name.split('_')[1]
    # img_index = source_dataset['image_name'].index(source_image_name)
    # reference_image, _ = get_image_and_name(source_dataset, img_index)
        
    return word_image, image_name, "", typed_word, image_name, image_index

def next_image(current_index, typed_word, typed_words, image_names):
    print("Next button clicked")
    print(current_index)
    new_index = current_index + 1
    if new_index >= len(word_dataset["train"]):
        new_index = 0
    return update_image(new_index, typed_word, typed_words, image_names)

def previous_image(current_index, typed_word, typed_words, image_names):
    new_index = current_index -1
    if new_index < 0:
        new_index = len(word_dataset["train"]) - 1
    return update_image(new_index, typed_word, typed_words, image_names)

# Create a Gradio interface
with gr.Blocks() as demo:
    word_display = gr.Image(type="pil", height=224, width=224)
    # source_display = gr.Image(type="pil", height=224, width=224)
    image_name_display = gr.Textbox(label="Image Name")
    word_input = gr.Textbox(label="Type the word you see in the image")
    current_index = gr.Number(value=index, label="Image Index", interactive=False)
    typed_words = gr.State([])
    image_names = gr.State([])

    next_button = gr.Button("Next")
    prev_button = gr.Button("Previous")

    prev_button.click(previous_image, inputs=[current_index, word_input, typed_words, image_names], outputs=[word_display, image_name_display, word_input, typed_words, image_names, current_index])
    next_button.click(next_image, inputs=[current_index, word_input, typed_words, image_names], outputs=[word_display, image_name_display, word_input, typed_words, image_names, current_index])

    # image_row = gr.Row(word_display, source_display)
    # Initialize the first image
    demo.load(update_image, inputs=[current_index, word_input, typed_words, image_names], outputs=[word_display, image_name_display, word_input, typed_words, image_names, current_index])

# Launch the app
demo.launch()
