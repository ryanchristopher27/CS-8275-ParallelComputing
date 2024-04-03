from PIL import Image

# Open the input .tif image
input_image = Image.open("../data/airplane00.tif")

# Get the original width and height of the image
original_width, original_height = input_image.size

# Calculate the new width and height (5 times larger)
new_width = original_width * 5
new_height = original_height * 5

# Resize the image using bilinear interpolation
output_image = input_image.resize((new_width, new_height), resample=Image.BILINEAR)

# Save the scaled image as a new .tif file
output_image.save("../data/scaled_airplane00.tif")

print("Image scaled successfully!")