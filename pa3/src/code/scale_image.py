from PIL import Image

# Open the input .tif image
input_image = Image.open("../data/airplane00.tif")

# Scale Image
original_width, original_height = input_image.size
new_width = original_width * 5
new_height = original_height * 5
output_image = input_image.resize((new_width, new_height), resample=Image.BILINEAR)

# Save the scaled image as a new .tif file
output_image.save("../data/scaled_airplane00.tif")

print("Image scaled successfully!")