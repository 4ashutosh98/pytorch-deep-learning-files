
import os
import zipfile

from pathlib import Path

import requests

def download_data(raw_url_to_dataset: str,
                  target_path:str = None,
                  remove_source: bool = True):
    """
    Download and extract a dataset from a specified URL to a target directory.

    Args:
        target_path (str, optional): The target directory where the data should be saved. 
                                     Defaults to the current working directory.
        raw_url_to_dataset (str): The URL of the dataset to be downloaded.

    Returns:
        Tuple[Path, Path]: Paths to the train and test directories.
    """
    # Setup the path to target folder
    if target_path == None:
        target_path = Path.cwd()
    # Setup path to data folder
    data_path = Path(target_path) / "data"
    image_path = data_path / "pizza_steak_sushi"

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory already exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
    # Download pizza, steak, sushi data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get(raw_url_to_dataset)
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...") 
        zip_ref.extractall(image_path)
        
    # Remove zip file
    if remove_source:
        os.remove(data_path / "pizza_steak_sushi.zip")

    # Creating the train and test directory path
    train_dir = image_path/"train"
    test_dir = image_path/"test"

    return train_dir, test_dir 
