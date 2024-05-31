import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage
import os

def create_csv(image_path:str):
    # Create a DataFrame with image paths and names
    data = {
        "image_path": [image_path + f for f in os.listdir(image_path) if not f.endswith('.csv') ],  # Replace with actual paths
        "image_name": [f for f in os.listdir(image_path) if not f.endswith('.csv')]
    }
    df = pd.DataFrame(data)

    # Save DataFrame to CSV
    df.to_csv(f"{image_path}image_dataset.csv", index=False)

    dataset = Dataset.from_csv(f"{image_path}image_dataset.csv")

    initial_features = Features({
        "image_path": Value("string"),
        "image_name": Value("string"),
    })

    # Apply the correct features to the dataset
    dataset = dataset.cast(initial_features)

    return dataset

def load_images(example):
    # Load the image from the path and add it to the example dictionary
    example["image"] = PILImage.open(example["image_path"])
    return example

# Apply the function to the dataset
def main(image_path:str, hf_path:str):
    dataset = create_csv(image_path)
    dataset = dataset.map(load_images)

    final_features = Features({
        "image_name": Value("string"),
        "image_path": Value("string"),
        "image": Image()
    })
    dataset = dataset.cast(final_features)

    # Drop the 'image_path' column if not needed anymore
    dataset = dataset.remove_columns(["image_path"])

    # Optional: Split the dataset if needed
    dataset_dict = DatasetDict({"train": dataset})
    print(dataset_dict)
    dataset_dict.push_to_hub(f"{hf_path}")

if __name__ == '__main__':
    main('refined_pages/', 'straka/source_pages')
