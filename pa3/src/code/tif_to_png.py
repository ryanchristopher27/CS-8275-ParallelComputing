from PIL import Image

# Open the TIFF file
tiff_image = Image.open("../data/airplane00.tif")

# Convert the TIFF image to RGB mode (if needed)
rgb_image = tiff_image.convert("RGB")

# Save the image as PNG
rgb_image.save("../data/airplane00.png", "PNG")

print("TIFF image converted to PNG successfully!")