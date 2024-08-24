"""
Utility functions to make prediction.
"""

import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from typing import List, Tuple

from PIL import Image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def pred_and_plot_image(model: torch.nn.Module,
                        class_names: List[str],
                        image_path : str,
                        image_size: Tuple[int,int] = (224,224),
                        transform : torchvision.transforms = None,
                        device: torch.device = device):
    
    """Makes a prediciton on a target image with a trained model and plots the image and prediction.
    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_path (str): Filepath to target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
    """

    # 2. Open the image
    custom_image = Image.open(image_path)

    # 3. Create a transform
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # 4. Moving the model on the device
    model.to(device)

    # 5. Setting the model to eval mode
    model.eval()
    with torch.inference_mode():
        # 6. Transform the target image
        custom_image_transformed = image_transform(custom_image).unsqueeze(dim =0) # [batch_size, color_channels, height, width]

        # 7. Making a prediciton on the image and also ensure that it is on the target device

        pred_logits = model(custom_image_transformed.to(device))

    # 8. converting logits to pred probs
    pred_probs = torch.softmax(pred_logits, dim = 1)

    # 9. Converting pred probs to pred labels
    pred_label = torch.argmax(pred_probs, dim = 1).cpu()

    # 10. plotting the image and with prediciton class and prediction probability
    plt.figure(figsize=(10,7))
    plt.imshow(custom_image)

    # 10.1 Obtain the class of the image
    if class_names:
        title = f"Pred: {class_names[pred_label.cpu()]} | Prob: {pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {pred_label.cpu()} | Prob: {pred_probs.max().cpu():.3f}"

    plt.title(title,fontsize = 20)
    plt.axis(False)
