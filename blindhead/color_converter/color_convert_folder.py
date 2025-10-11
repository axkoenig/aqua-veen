import argparse
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


def convert_folder(
    input_dir, output_dir, background_color, positive_color, negative_color, suffix
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            file_path = os.path.join(input_dir, filename)
            image_greyscale = read_image(file_path)

            converted_image = convert_image(
                image_greyscale, background_color, positive_color, negative_color
            )
            output_file = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}{suffix}.tiff"
            )
            converted_image.save(output_file, "TIFF", compression="tiff_lzw")

            print(f"Processed and saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert images with a custom color scheme."
    )
    parser.add_argument("--input_dir", help="Directory containing input images.")
    parser.add_argument("--output_dir", help="Directory to save converted images.")
    parser.add_argument(
        "--background_color",
        default="#1f2436",
        help="Background color in hex format.",
    )
    parser.add_argument(
        "--positive_color",
        default="#fb3737",
        help="Positive color in hex format.",
    )
    parser.add_argument(
        "--negative_color",
        default="#fdf2f2",
        help="Negative color in hex format.",
    )
    parser.add_argument(
        "--suffix",
        default="_converted",
        help="Suffix to add to the converted file names.",
    )

    args = parser.parse_args()

    convert_folder(
        args.input_dir,
        args.output_dir,
        args.background_color,
        args.positive_color,
        args.negative_color,
        args.suffix,
    )


if __name__ == "__main__":
    main()
