# Imports
import torch
from PIL import Image
import numpy as np

def cuda_setup(d='gpu') -> tuple:
    if d == 'gpu':
        if torch.cuda.is_available():
            print(torch.cuda.current_device())     # The ID of the current GPU.
            print(torch.cuda.get_device_name(id))  # The name of the specified GPU, where id is an integer.
            print(torch.cuda.device(id))           # The memory address of the specified GPU, where id is an integer.
            print(torch.cuda.device_count())
            
        on_gpu = torch.cuda.is_available()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Device: {device}')
    
    else:
        on_gpu = False
        device = torch.device("cpu")
        print(f'Device: {device}')

    return device, on_gpu


def image_to_tensor(image: Image, device: torch.device, on_gpu: bool):

    # Open the TIFF file
    # image = Image.open("../data/airplane00.tif")

    # Display some information about the image
    # print("Image format:", image.format)
    # print("Image size:", image.size)
    # print("Image mode:", image.mode)

    # Display the image
    # image.show()

    image_np = np.array(image)

    input_tensor = torch.from_numpy(image_np)
    input_tensor = input_tensor.float()
    input_tensor = input_tensor / 255.0

    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)

    input_tensor = input_tensor.to(device)

    return input_tensor


def tensor_to_image(output_tensor: torch.Tensor):

    output_tensor = output_tensor.cpu()

    output_tensor = output_tensor.squeeze(0).permute(1, 2, 0)
    output_tensor = output_tensor * 255.0

    output_image = Image.fromarray(output_tensor.numpy().astype(np.uint8))

    return output_image


# def data_preprocessing(image: any, input_type: str, output_type: str):
#     if input_type == 'tif':

