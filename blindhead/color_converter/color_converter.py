import json
import os

import cv2
import streamlit as st


@st.cache_data
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


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def save_config(config: dict, save_path: str) -> None:
    with open(save_path, "w") as f:
        json.dump(config, f)


def main(
    data_dir: str = "/Users/koenig/Documents/personal/art projects/colab justin/blindhead/proc/events_for_art/color_converter/input",
    output_dir: str = "/Users/koenig/Documents/personal/art projects/colab justin/blindhead/proc/events_for_art/color_converter/output",
) -> None:

    with st.sidebar:
        st.title("Color Converter")
        st.header("Settings")

        if not os.path.exists(data_dir):
            st.error("Invalid directory path")
            return

        os.makedirs(output_dir, exist_ok=True)
        # list all available configs in the output directory
        available_configs = [f for f in os.listdir(output_dir) if f.endswith(".json")]
        config = {}
        if available_configs:
            selected_config = st.selectbox("Select a config", available_configs)
            config = load_config(os.path.join(output_dir, selected_config))

        all_files = os.listdir(data_dir)
        selected_file = st.selectbox("Select a file", all_files)

        background_color = st.color_picker(
            "Background color", value=config.get("background_color", "#000000")
        )
        positive_color = st.color_picker(
            "Positive color", value=config.get("positive_color", "#1111aa")
        )
        negative_color = st.color_picker(
            "Negative color", value=config.get("negative_color", "#FF0000")
        )

    image_greyscale = read_image(os.path.join(data_dir, selected_file))
    converted_image = convert_image(
        image_greyscale, background_color, positive_color, negative_color
    )

    st.image(converted_image, caption="Converted Image", use_container_width=True)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, selected_file)
    if st.button("Save Image"):
        cv2.imwrite(save_path, converted_image.astype("uint8")[:, :, ::-1])
        st.success(f"Image saved to {save_path}")
        colors = {
            "background_color": background_color,
            "positive_color": positive_color,
            "negative_color": negative_color,
        }
        save_config(colors, os.path.join(output_dir, "colors.json"))


main()
