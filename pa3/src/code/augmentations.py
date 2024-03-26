# Imports
from kornia import augmentation
import torch
import cupy as cp
import cupyx.scipy.ndimage as ndimage

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



def get_cupy_augmentation(augmentation_name, degrees=90, mean=0.0, std=0.1):
    if augmentation_name == 'RandomHorizontalFlip':
        return lambda x: x[:, ::-1] # flip horizontally
    elif augmentation_name == 'RandomVerticalFlip':
        return lambda x: x[::-1, :] # flip vertically
    elif augmentation_name == 'RandomRotation':
        return lambda x: ndimage.rotate(x, degrees, reshape=False)
    elif augmentation_name == 'RandomGaussianNoise':
        return lambda x: cp.clip(cp.add(x, cp.random.normal(mean, std*255, x.shape)), 0, 255)
    else:
        return ValueError(f"Invalid Augmentation Name of {augmentation_name}")




