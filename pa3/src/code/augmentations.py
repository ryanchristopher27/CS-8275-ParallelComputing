# Imports
from kornia import augmentation
import torch

def get_kornia_augmentation(augmentation_name, p=1.0, degrees=[90.0, 90.0], mean=0.0, std=0.1):
    if augmentation_name == 'RandomHorizontalFlip':
        return augmentation.RandomHorizontalFlip(p=p)

    elif augmentation_name == 'RandomVerticalFlip':
        return augmentation.RandomVerticalFlip(p=p)
    
    elif augmentation_name == 'RandomRotation':
        return augmentation.RandomRotation(degrees=degrees, p=p)

    elif augmentation_name == 'RandomGaussianNoise':
        return augmentation.RandomGaussianNoise(mean=mean, std=std, p=p)
    else:
        return ValueError(f"Invalid Augmentation Name of {augmentation_name}")




# def perform_augmentation(image, start :_CudaEventBase, end):

#     seq = get_augmentation(augmentation_name=augmentation_name, p=p)

#     output = 
