import argparse
import json
import os

from PIL import Image


def read_image(image_path: str) -> Image:
    return Image.open(image_path).convert("L")  # Open and convert to grayscale


def convert_image(
    image_greyscale: Image,
    background_color: str,
    positive_color: str,
    negative_color: str,
    background_max_int=70,
    positive_min_int=200,
) -> Image:
    # Convert grayscale image to RGB
    image_rgb = image_greyscale.convert("RGB")
    pixels = image_rgb.load()

    # Convert hex colors to RGB tuples
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)

    background_color_rgb = hex_to_rgb(background_color)
    positive_color_rgb = hex_to_rgb(positive_color)
    negative_color_rgb = hex_to_rgb(negative_color)

    # Modify pixels based on grayscale values
    width, height = image_rgb.size
    for i in range(width):
        for j in range(height):
            pixel_value = image_greyscale.getpixel((i, j))
            if 0 <= pixel_value <= background_max_int:
                pixels[i, j] = background_color_rgb
            elif background_max_int < pixel_value <= positive_min_int:
                pixels[i, j] = positive_color_rgb
            elif pixel_value >= positive_min_int:
                pixels[i, j] = negative_color_rgb

    return image_rgb


def convert_image_with_config(image_path, output_dir, config_path):
    # Read config file
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    # Read image
    image_greyscale = read_image(image_path)

    # Extract color values from config
    background_color = config.get("background_color", "#1f2436")
    positive_color = config.get("positive_color", "#fb3737")
    negative_color = config.get("negative_color", "#fdf2f2")

    # Convert the image
    converted_image = convert_image(
        image_greyscale, background_color, positive_color, negative_color
    )

    # Use the config name (without extension) as the output filename
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    output_file = os.path.join(output_dir, f"{config_name}.tiff")

    # Save the converted image
    converted_image.save(output_file, "TIFF", compression="tiff_lzw")
    print(f"Processed and saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert images with a custom color scheme from JSON config files."
    )

    output_dir= "/Users/koenig/Documents/personal/art_projects/colab_justin/blindhead/events_for_art/color_converter/output_rerender"

    image_path = "/Users/koenig/Documents/personal/art_projects/colab_justin/blindhead/events_for_art/color_converter/input/recording_2025-01-10_17-00-07.avi.00_08_55_11.Standbild002.jpg"
    config_paths = [
        "/Users/koenig/Documents/personal/art_projects/colab_justin/blindhead/events_for_art/color_converter/random_color_schemes_batch_2_100_imgs/configs/converted_image_19.json",
        "/Users/koenig/Documents/personal/art_projects/colab_justin/blindhead/events_for_art/color_converter/random_color_schemes_batch_1_100_imgs/configs/converted_image_13.json",
        "/Users/koenig/Documents/personal/art_projects/colab_justin/blindhead/events_for_art/color_converter/random_color_schemes_batch_3_1000_imgs/configs/converted_image_77.json",
        "/Users/koenig/Documents/personal/art_projects/colab_justin/blindhead/events_for_art/color_converter/random_color_schemes_batch_3_1000_imgs/configs/converted_image_7.json",
    ]

    args = parser.parse_args()

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process the image with each config
    for config_path in config_paths:
        convert_image_with_config(image_path, output_dir, config_path)


if __name__ == "__main__":
    main()
