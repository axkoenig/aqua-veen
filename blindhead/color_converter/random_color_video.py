import argparse
import json
import os
import random

import cv2
import numpy as np
from tqdm import tqdm  # Importing tqdm for the progress bar


def generate_random_color_scheme(min_brightness_diff=20):
    def random_color():
        return "#" + "".join(random.choices("0123456789ABCDEF", k=6))

    def brightness(color):
        r, g, b = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
        return 0.299 * r + 0.587 * g + 0.114 * b

    while True:
        bg_color = random_color()
        pos_color = random_color()
        neg_color = random_color()

        if (
            abs(brightness(bg_color) - brightness(pos_color)) >= min_brightness_diff
            and abs(brightness(pos_color) - brightness(neg_color))
            >= min_brightness_diff
            and abs(brightness(bg_color) - brightness(neg_color)) >= min_brightness_diff
        ):
            return bg_color, pos_color, neg_color


def convert_image(
    image_greyscale: np.ndarray,
    background_color: str,
    positive_color: str,
    negative_color: str,
    background_max_int=70,
    positive_min_int=200,
) -> np.ndarray:

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


def read_image(image_path: str) -> np.ndarray:
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def generate_video(
    input_file: str,
    video_length: int = 60,
    frames_per_image: int = 1,
    output_file: str = "output_video.mp4",
):
    if not os.path.exists(input_file):
        print("Input file does not exist.")
        return

    image_greyscale = read_image(input_file)

    # Generate random color schemes and images
    num_frames = video_length * frames_per_image
    frames = []

    # Using tqdm for progress bar while generating frames
    for i in tqdm(range(num_frames), desc="Generating frames", unit="frame"):
        background_color, positive_color, negative_color = (
            generate_random_color_scheme()
        )

        converted_image = convert_image(
            image_greyscale, background_color, positive_color, negative_color
        )

        frames.extend(
            [converted_image] * frames_per_image
        )  # Repeat the frame for frames_per_image

    # Get the dimensions of the image for the video
    height, width, _ = frames[0].shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_file, fourcc, 30, (width, height))

    # Writing frames to video with progress bar
    for frame in tqdm(frames, desc="Writing video", unit="frame"):
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from randomly colored images."
    )

    # Input file
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input grayscale image.",
        default="/Users/koenig/Documents/personal/art_projects/colab_justin/blindhead/events_for_art/color_converter/input/recording_2025-01-10_17-00-07.avi.00_08_55_11.Standbild002.jpg",
    )

    # Output video file
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the output video file.",
        default="/Users/koenig/Documents/personal/art_projects/colab_justin/blindhead/events_for_art/color_converter/random_video_out/recording_2025-01-10_17-00-07.avi.00_08_55_11.Standbild002.mp4",
    )

    # Video length (in seconds)
    parser.add_argument(
        "--video_length",
        type=int,
        default=120,
        help="Length of the video in seconds (default: 60).",
    )

    # Frames per image
    parser.add_argument(
        "--frames_per_image",
        type=int,
        default=1,
        help="Number of frames each generated image should be shown for (default: 1).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments from command line
    args = parse_args()

    # Generate the video
    generate_video(
        input_file=args.input_file,
        video_length=args.video_length,
        frames_per_image=args.frames_per_image,
        output_file=args.output_file,
    )
