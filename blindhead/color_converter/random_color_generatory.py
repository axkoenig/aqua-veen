import json
import os
import random

import cv2
import streamlit as st


def read_image(image_path: str) -> cv2.imread:
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def convert_image(
    image_greyscale: cv2.imread,
    background_color: str,
    positive_color: str,
    negative_color: str,
    background_max_int=70,
    positive_min_int=200,
) -> cv2.imread:

    # Convert grayscale to BGR
    image = cv2.cvtColor(image_greyscale, cv2.COLOR_GRAY2BGR)

    # Convert hex colors to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")  # Remove '#' if present
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)

    background_color_rgb = hex_to_rgb(background_color)
    positive_color_rgb = hex_to_rgb(positive_color)
    negative_color_rgb = hex_to_rgb(negative_color)

    # Replace all pixels in the background range with the background color
    background_mask = (image_greyscale >= 0) & (image_greyscale <= background_max_int)
    image[background_mask] = background_color_rgb[::-1]  # Convert RGB to BGR for OpenCV

    # Replace all pixels in the positive range with the positive color
    positive_mask = (image_greyscale > background_max_int) & (
        image_greyscale < positive_min_int
    )
    image[positive_mask] = positive_color_rgb[::-1]  # Convert RGB to BGR for OpenCV

    # Replace all pixels in the negative range with the negative color
    negative_mask = image_greyscale >= positive_min_int
    image[negative_mask] = negative_color_rgb[::-1]  # Convert RGB to BGR for OpenCV

    # Convert the image back to RGB format for rendering
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb

def save_config(config: dict, save_path: str) -> None:
    with open(save_path, "w") as f:
        json.dump(config, f, indent=4)

def generate_random_color_scheme(min_brightness_diff=20):
    def random_color():
        return "#" + "".join(random.choices("0123456789ABCDEF", k=6))

    def brightness(color):
        r, g, b = [int(color[i:i+2], 16) for i in (1, 3, 5)]
        return 0.299 * r + 0.587 * g + 0.114 * b

    while True:
        bg_color = random_color()
        pos_color = random_color()
        neg_color = random_color()
        
        if (
            abs(brightness(bg_color) - brightness(pos_color)) >= min_brightness_diff
            and abs(brightness(pos_color) - brightness(neg_color)) >= min_brightness_diff
            and abs(brightness(bg_color) - brightness(neg_color)) >= min_brightness_diff
        ):
            return bg_color, pos_color, neg_color

def main(
    input_file: str = "/Users/koenig/Documents/personal/art projects/colab justin/blindhead/proc/events_for_art/color_converter/input/recording_2025-01-10_17-00-07.avi.00_08_55_11.Standbild002.jpg",
    output_dir: str = "/Users/koenig/Documents/personal/art projects/colab justin/blindhead/proc/events_for_art/color_converter/random_color_schemes_batch_4_1000_imgs",
    num_samples: int = 100,
):
    if not os.path.exists(input_file):
        print("Input file does not exist.")
        return

    images_dir = os.path.join(output_dir, "images")
    configs_dir = os.path.join(output_dir, "configs")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)

    image_greyscale = read_image(input_file)

    for i in range(num_samples):
        background_color, positive_color, negative_color = generate_random_color_scheme()

        converted_image = convert_image(
            image_greyscale, background_color, positive_color, negative_color
        )

        output_image_path = os.path.join(images_dir, f"converted_image_{i + 1}.tiff")
        output_config_path = os.path.join(configs_dir, f"converted_image_{i + 1}.json")

        cv2.imwrite(output_image_path, converted_image.astype("uint8")[:, :, ::-1])

        color_config = {
            "background_color": background_color,
            "positive_color": positive_color,
            "negative_color": negative_color,
        }
        save_config(color_config, output_config_path)

    print(f"{num_samples} images have been saved to {images_dir} and configurations to {configs_dir}.")

if __name__ == "__main__":
    main()
