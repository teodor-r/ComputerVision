import torch
from torch.utils.data import Subset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def tensor_to_image(tensor, denormalize=True):
    image = tensor.clone().detach().cpu()
    if image.dim() == 4:
        image = image[0]
    image = image.permute(1, 2, 0)

    if denormalize:
        # Standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image = image * std + mean

    image = image.numpy()
    image = np.clip(image, 0, 1)

    return image


def show_tensor_as_image(tensor):
    image = tensor_to_image(tensor)

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Matplotlib Display')
    plt.show()

    return image