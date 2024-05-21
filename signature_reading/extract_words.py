import os
from PIL import Image
import pillow_heif
import cv2
import numpy as np

def convert_heic_to_jpeg(heic_path, jpeg_path):
    heif_image = pillow_heif.read_heif(heic_path)
    image = Image.frombytes(
        heif_image.mode, 
        heif_image.size, 
        heif_image.data,
        "raw",
        heif_image.mode,
        heif_image.stride,
    )
    image.save(jpeg_path, "JPEG")

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    return image, binary

def extract_words(binary_image):
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    return contours

def resize_image(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        # Calculate the ratio of the width and construct the dimensions
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    
    # Resize the image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def visualize_bounding_boxes(image, contours):
    # Draw bounding boxes on the original image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 50 < w < 1200 and h > 20:  # Adjust these parameters as needed
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def save_word_images(image, contours, output_dir, base_file_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    word_images = []
    
    for i, contour in enumerate(contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out contours that are too wide (likely sentences/lines)
        if 50 < w < 1200 and h > 20:  # Adjust 300 based on your observations
            word_image = image[y:y+h, x:x+w]
            word_images.append(word_image)
            
            # Save the image with a meaningful name
            word_img_path = os.path.join(output_dir, f'{base_file_name}_word_{i+1}.png')
            cv2.imwrite(word_img_path, word_image)
    
    return word_images


def main(heic_path, output_dir):
    # Convert HEIC to JPEG
    file_name = os.path.basename(heic_path).split(".")[0]
    print("file_name", heic_path, "and", file_name)
    jpeg_path = f"./refined_pages/{file_name}.jpg"
    convert_heic_to_jpeg(heic_path, jpeg_path)
    
    # Preprocess the image
    original_image, binary_image = preprocess_image(jpeg_path)
    
    # Extract words
    contours = extract_words(binary_image)

    # Visualize bounding boxes
    visualized_image = visualize_bounding_boxes(original_image.copy(), contours)
    
    # Resize image for display
    resized_image = resize_image(visualized_image, width=1000)  # Adjust width as needed
    
    # Display the image with bounding boxes
    cv2.imshow('Bounding Boxes', resized_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    
    # Save word images
    word_images = save_word_images(original_image, contours, output_dir, file_name)
    
    print(f"Extracted and saved {len(word_images)} words.")

if __name__ == "__main__":
    heic_path = "raw_pages/IMG_9829.HEIC"  # Replace with your HEIC image path
    output_dir = "word_images"  # Directory to save word images
    
    main(heic_path, output_dir)
