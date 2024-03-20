from PIL import Image
import torch
from kornia import augmentation
import numpy as np
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm


from augmentations import *
from utils import *

def main():

    # augmentation_experiment()
    # cupy_test()
    # single_augmentation_exp('RandomRotation') # 'RandomRotation'
    cifar10_augmentation_exp('RandomRotation')

def augmentation_experiment():
    device, on_gpu = cuda_setup('gpu')
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

    # image = Image.open("../data/airplane00.tif")
    # input_tensor = tif_to_tensor(image, device, on_gpu) 

    seq = get_kornia_augmentation(augmentation_name='RandomGaussianNoise', p=1.0)
    start_event.record()
    output = seq(input_tensor)
    end_event.record()
    torch.cuda.synchronize() 

    output_image = tensor_to_image(output)
    output_image.show()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Augmentation time on GPU: {elapsed_time_ms:.2f} ms")


def single_augmentation_exp(augmentation_type :str):
    device, on_gpu = cuda_setup('cpu')

    input_image = Image.open("../data/airplane00.tif")
    input_tensor = image_to_tensor(input_image, device, on_gpu) 

    # ======================================================================
    # Kornia Section
    # ======================================================================
    start_kornia = torch.cuda.Event(enable_timing=True)
    end_kornia = torch.cuda.Event(enable_timing=True)

    seq = get_kornia_augmentation(augmentation_name=augmentation_type)
    start_kornia.record()
    output_kornia = seq(input_tensor)
    end_kornia.record()
    torch.cuda.synchronize() 

    kornia_output = tensor_to_image(output_kornia)

    kornia_time = start_kornia.elapsed_time(end_kornia)
    print(f"Kornia time on GPU: {kornia_time:.2f} ms")

    # ======================================================================
    # Cupy Section
    # ======================================================================
    start_cupy = torch.cuda.Event(enable_timing=True)
    end_cupy = torch.cuda.Event(enable_timing=True)

    cupy_image = cp.asarray(input_image)
    angle = 90
    start_cupy.record()
    output_cupy = ndimage.rotate(cupy_image, angle, reshape=False)
    end_cupy.record()
    torch.cuda.synchronize() 

    output_np_cupy = cp.asnumpy(output_cupy)
    cupy_output = Image.fromarray(output_np_cupy)

    cupy_time = start_cupy.elapsed_time(end_cupy)
    print(f"Cupy time on GPU: {cupy_time:.2f} ms")

    # ======================================================================
    # Display Section
    # ======================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(kornia_output)
    ax1.set_title("Kornia Output")
    ax1.axis("off")

    ax2.imshow(cupy_output)
    ax2.set_title("CuPy Output")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


def cifar10_augmentation_exp(augmentation_type: str):
    device, on_gpu = cuda_setup('gpu')

    # Define the preprocessing transforms
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the CIFAR-10 dataset
    dataset = datasets.CIFAR10(root='../data/cifar10', train=True, transform=preprocess, download=True)

    # Initialize variables to store total times
    total_kornia_time = 0
    total_cupy_time = 0

    # Iterate over the dataset
    for i in tqdm(range(len(dataset)), desc="Processing images"):
        input_image, _ = dataset[i]
        input_tensor = input_image.unsqueeze(0).to(device)

        # ======================================================================
        # Kornia Section
        # ======================================================================
        start_kornia = torch.cuda.Event(enable_timing=True)
        end_kornia = torch.cuda.Event(enable_timing=True)

        seq = get_kornia_augmentation(augmentation_name=augmentation_type)
        start_kornia.record()
        output_kornia = seq(input_tensor)
        end_kornia.record()
        torch.cuda.synchronize()

        kornia_time = start_kornia.elapsed_time(end_kornia)
        total_kornia_time += kornia_time

        # ======================================================================
        # Cupy Section
        # ======================================================================
        start_cupy = torch.cuda.Event(enable_timing=True)
        end_cupy = torch.cuda.Event(enable_timing=True)

        cupy_image = cp.asarray(input_image.permute(1, 2, 0).numpy())
        angle = 90
        start_cupy.record()
        output_cupy = ndimage.rotate(cupy_image, angle, reshape=False)
        end_cupy.record()
        torch.cuda.synchronize()

        cupy_time = start_cupy.elapsed_time(end_cupy)
        total_cupy_time += cupy_time

    # Calculate average times
    avg_kornia_time = total_kornia_time / len(dataset)
    avg_cupy_time = total_cupy_time / len(dataset)
    print(f"Average Kornia time on GPU: {avg_kornia_time:.4f} ms")
    print(f"Average Cupy time on GPU: {avg_cupy_time:.4f} ms\n")

    total_kornia_minutes, total_kornia_seconds = divmod(total_kornia_time / 1000, 60)
    total_kornia_milliseconds = total_kornia_time % 1000
    total_cupy_minutes, total_cupy_seconds = divmod(total_cupy_time / 1000, 60)
    total_cupy_milliseconds = total_cupy_time % 1000
    print(f"Total Kornia time on GPU: {int(total_kornia_minutes):02d}:{int(total_kornia_seconds):02d}:{int(total_kornia_milliseconds):03d}")
    print(f"Total Cupy time on GPU: {int(total_cupy_minutes):02d}:{int(total_cupy_seconds):02d}:{int(total_cupy_milliseconds):03d}\n")


def cupy_test():
    # Load the image using PIL
    image = Image.open("../data/airplane00.tif")

    # Convert the image to a CuPy array
    cupy_image = cp.asarray(image)

    # Specify the rotation angle in degrees
    angle = 90

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    # Perform the rotation
    rotated_image = ndimage.rotate(cupy_image, angle, reshape=False)
    end_event.record()

    # Convert the rotated image back to a NumPy array
    rotated_image_np = cp.asnumpy(rotated_image)

    # Convert the NumPy array back to a PIL image
    rotated_image_pil = Image.fromarray(rotated_image_np)

    # Display the rotated image
    rotated_image_pil.show()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Augmentation time on GPU: {elapsed_time_ms:.2f} ms")


if __name__ == "__main__":
    main()