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
    augs = ['RandomRotation', 'RandomVerticalFlip', 'RandomHorizontalFlip', 'RandomGaussianNoise']
    # augs = ['RandomVerticalFlip', 'RandomHorizontalFlip', 'RandomGaussianNoise']
    # single_augmentation_exp('RandomGaussianNoise') 
    # single_augmentation_n_runs_exp('RandomGaussianNoise', 100)
    # cifar10_augmentation_exp('RandomRotation')

    for aug in augs:
        print("==="*20 + f"\n{aug}\n" + "==="*20)
        # single_augmentation_exp(aug)
        # single_augmentation_n_runs_exp(aug, 100)
        cifar10_augmentation_exp(aug)

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


def single_augmentation_exp(augmentation_type: str):
    device, on_gpu = cuda_setup('gpu')
    input_image = Image.open("../data/airplane00.tif")
    input_tensor = image_to_tensor(input_image, device, on_gpu)
    input_numpy = np.array(input_image)

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
    print(f"Kornia time on GPU: {kornia_time:.4f} ms")

    # ======================================================================
    # Cupy Section
    # ======================================================================
    start_cupy = torch.cuda.Event(enable_timing=True)
    end_cupy = torch.cuda.Event(enable_timing=True)
    cupy_image = cp.asarray(input_image)
    seq = get_cupy_augmentation(augmentation_name=augmentation_type)
    start_cupy.record()
    output_cupy = seq(cupy_image)
    end_cupy.record()
    torch.cuda.synchronize()
    output_np_cupy = cp.asnumpy(output_cupy).astype(np.uint8)
    cupy_output = Image.fromarray(output_np_cupy)
    cupy_time = start_cupy.elapsed_time(end_cupy)
    print(f"Cupy time on GPU: {cupy_time:.4f} ms")

    # ======================================================================
    # CPU Section
    # ======================================================================
    seq = get_cpu_augmentation(augmentation_name=augmentation_type)
    start_numpy = time.perf_counter()
    output_numpy = seq(input_numpy)
    end_numpy = time.perf_counter()
    numpy_output = Image.fromarray(output_numpy.astype(np.uint8))
    numpy_time = (end_numpy - start_numpy) * 1000  # Convert to milliseconds
    print(f"NumPy time on CPU: {numpy_time:.4f} ms")

    # ======================================================================
    # Display Section
    # ======================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(kornia_output)
    ax1.set_title("Kornia Output")
    ax1.axis("off")
    ax2.imshow(cupy_output)
    ax2.set_title("CuPy Output")
    ax2.axis("off")
    ax3.imshow(numpy_output)
    ax3.set_title("CPU Output")
    ax3.axis("off")
    plt.tight_layout()
    plt.show()


def single_augmentation_n_runs_exp(augmentation_type :str, num_runs :int):
    device, on_gpu = cuda_setup('gpu')

    input_image = Image.open("../data/airplane00.tif")
    input_tensor = image_to_tensor(input_image, device, on_gpu) 
    input_numpy = np.array(input_image)

    total_kornia_time = 0
    total_cupy_time = 0
    total_numpy_time = 0

    for i in tqdm(range(num_runs), desc="Processing Images"):
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

        cupy_image = cp.asarray(input_image)
        angle = 90
        seq = get_cupy_augmentation(augmentation_name=augmentation_type)
        start_cupy.record()
        output_cupy = seq(cupy_image)
        end_cupy.record()
        torch.cuda.synchronize() 

        cupy_time = start_cupy.elapsed_time(end_cupy)
        total_cupy_time += cupy_time

        # ======================================================================
        # CPU Section
        # ======================================================================
        seq = get_cpu_augmentation(augmentation_name=augmentation_type)
        start_numpy = time.perf_counter()
        output_numpy = seq(input_numpy)
        end_numpy = time.perf_counter()
        numpy_time = (end_numpy - start_numpy) * 1000  # Convert to milliseconds
        total_numpy_time += numpy_time

    # Calculate average times
    avg_kornia_time = total_kornia_time / num_runs
    avg_cupy_time = total_cupy_time / num_runs
    avg_numpy_time = total_numpy_time / num_runs

    print(f"Average Kornia time on GPU: {avg_kornia_time:.4f} ms")
    print(f"Average Cupy time on GPU: {avg_cupy_time:.4f} ms\n")
    print(f"Average NumPy time on CPU: {avg_numpy_time:.4f} ms\n")


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
    total_numpy_time = 0

    # ======================================================================
    # Kornia Section
    # ======================================================================
    start_kornia = torch.cuda.Event(enable_timing=True)
    end_kornia = torch.cuda.Event(enable_timing=True)

    seq = get_kornia_augmentation(augmentation_name=augmentation_type)
    start_kornia.record()
    for i in tqdm(range(len(dataset)), desc="Processing images"):
        input_image, _ = dataset[i]
        input_tensor = input_image.unsqueeze(0).to(device)
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

    seq = get_cupy_augmentation(augmentation_name=augmentation_type)
    start_cupy.record()
    for i in tqdm(range(len(dataset)), desc="CuPy Processing"):
        input_image, _ = dataset[i]
        input_tensor = input_image.unsqueeze(0).to(device)
        cupy_image = cp.asarray(input_image.permute(1, 2, 0).numpy())
        output_cupy = seq(cupy_image)
    end_cupy.record()
    torch.cuda.synchronize()

    cupy_time = start_cupy.elapsed_time(end_cupy)
    total_cupy_time += cupy_time

    # ======================================================================
    # CPU Section
    # ======================================================================
    seq = get_cpu_augmentation(augmentation_name=augmentation_type)
    start_numpy = time.perf_counter()
    for i in tqdm(range(len(dataset)), desc="CPU Processing"):
        input_image, _ = dataset[i]
        input_numpy = np.array(input_image)
        output_numpy = seq(input_numpy)
    end_numpy = time.perf_counter()
    numpy_time = (end_numpy - start_numpy) * 1000  # Convert to milliseconds
    total_numpy_time += numpy_time

    # Calculate average times
    avg_kornia_time = total_kornia_time / len(dataset)
    total_kornia_minutes, total_kornia_seconds = divmod(total_kornia_time / 1000, 60)
    total_kornia_milliseconds = total_kornia_time % 1000
    print(f"Average Kornia time on GPU: {avg_kornia_time:.4f} ms")
    print(f"Total Kornia time on GPU: {int(total_kornia_minutes):02d}:{int(total_kornia_seconds):02d}:{int(total_kornia_milliseconds):03d}")
    
    avg_cupy_time = total_cupy_time / len(dataset)
    total_cupy_minutes, total_cupy_seconds = divmod(total_cupy_time / 1000, 60)
    total_cupy_milliseconds = total_cupy_time % 1000
    print(f"Average Cupy time on GPU: {avg_cupy_time:.4f} ms\n")
    print(f"Total Cupy time on GPU: {int(total_cupy_minutes):02d}:{int(total_cupy_seconds):02d}:{int(total_cupy_milliseconds):03d}\n")

    avg_numpy_time = total_numpy_time / len(dataset)
    total_numpy_minutes, total_numpy_seconds = divmod(total_numpy_time / 1000, 60)
    total_numpy_milliseconds = total_numpy_time % 1000
    print(f"Average Numpy time on GPU: {avg_numpy_time:.4f} ms\n")
    print(f"Total Numpy time on GPU: {int(total_numpy_minutes):02d}:{int(total_numpy_seconds):02d}:{int(total_numpy_milliseconds):03d}\n")


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