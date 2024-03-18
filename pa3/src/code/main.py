from PIL import Image
import torch
from kornia import augmentation
import numpy as np
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from augmentations import *
from utils import *

def main():
    device, on_gpu = cuda_setup('cpu')
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.CIFAR10(root='../data/cifar10', train=True, transform=preprocess, download=True)

    img, smnt = dataset[0]

    # Apply preprocessing transforms to the image
    input_tensor = img.to(device)
    # input_tensor = tif_to_tensor("../data/airplane00.tif", device, on_gpu) 

    seq = get_augmentation(augmentation_name='RandomGaussianNoise', p=1.0)
    start_event.record()
    output = seq(input_tensor)
    end_event.record()
    torch.cuda.synchronize() 

    output_pil = tensor_to_tif(output)
    output_pil.show()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Augmentation time on GPU: {elapsed_time_ms:.2f} ms")


if __name__ == "__main__":
    main()