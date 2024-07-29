"""
Contains functionality for creating PyTorch DataLoaders for image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 0

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir : Path to training directory.
        test_dir: Path to testing directory.
        train_transform: torchvision transforms to perform on training data.
        test_transform: torchvision transforms to perform on testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for the number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example Usage:
        train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir = path/to/train_sir,
            test_dir = path/to/test_dir,
            transform = some_transform,
            batch_size = 32,
            num_workers = 4)
    """

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(root = train_dir,
                                      transform = train_transform,
                                      target_transform = None)

    test_data = datasets.ImageFolder(root= test_dir,
                                     transform = test_transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into DataLoaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size = batch_size,
                                  num_workers = num_workers,
                                  shuffle = True,
                                  pin_memory = True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size = batch_size,
                                 num_workers = num_workers,
                                 shuffle = False,
                                 pin_memory = True)


    return train_dataloader, test_dataloader, class_names
