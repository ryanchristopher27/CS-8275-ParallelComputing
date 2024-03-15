from PIL import Image
import torch
from kornia import augmentation
import numpy as np

# Open the TIFF file
image = Image.open("../data/airplane00.tif")

# Display some information about the image
print("Image format:", image.format)
print("Image size:", image.size)
print("Image mode:", image.mode)

# Display the image
image.show()

image_np = np.array(image)

input_tensor = torch.from_numpy(image_np)
input_tensor = input_tensor.float()
input_tensor = input_tensor / 255.0

input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)


# Horizontal Flip
# seq = augmentation.RandomHorizontalFlip(p=1.0)
# Vertical Flip
# seq = augmentation.RandomVerticalFlip(p=1.0)
# Rotation
# seq = augmentation.RandomRotation(degrees=[90.0, 90.0], p=1.0)
# Gaussian Noise
seq = augmentation.RandomGaussianNoise(mean=0.0, std=0.1, p=1.0)

output = seq(input_tensor)

output = output.squeeze(0).permute(1, 2, 0)
output = output * 255.0

output_pil = Image.fromarray(output.numpy().astype(np.uint8))
output_pil.show()