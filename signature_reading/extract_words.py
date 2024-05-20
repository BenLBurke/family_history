import os
import pyheif
from PIL import Image
import cv2
import numpy as np

def convert_heic_to_jpeg(heic_path, jpeg_path):
    heif_file = pyheif.read(heic_path)
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data, 
        "raw", 
        heif_file.mode, 
        heif_file.stride,
    )
    image.save(jpeg_path, "JPEG")

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return image, binary

def extract_words(binary_image):
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    return contours

def save_word_images(image, contours, parent_file_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    word_images = []
    
    for i, contour in enumerate(contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out small or irrelevant contours
        if w > 10 and h > 10:  # you can adjust these parameters
            word_image = image[y:y+h, x:x+w]
            word_images.append(word_image)
            
            # Save the image
            word_img_path = os.path.join(output_dir, f'{parent_file_name}_word_{i+1}.png')
            cv2.imwrite(word_img_path, word_image)
    
    return word_images

def main(heic_path, output_dir):
    # Convert HEIC to JPEG
    file_name = heic_path.split(".")[0]
    jpeg_path = f"../refined_pages/{file_name}.jpg"
    convert_heic_to_jpeg(heic_path, jpeg_path)
    
    # Preprocess the image
    original_image, binary_image = preprocess_image(jpeg_path)
    
    # Extract words
    contours = extract_words(binary_image)
    
    # Save word images
    word_images = save_word_images(original_image, contours, file_name, output_dir)
    
    print(f"Extracted and saved {len(word_images)} words.")

if __name__ == "__main__":
    heic_path = "IMG_9829.HEIC"  # Replace with your HEIC image path
    output_dir = "word_images"  # Directory to save word images
    
    main(heic_path, output_dir)
