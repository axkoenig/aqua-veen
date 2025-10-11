import argparse

import cv2
import matplotlib.pyplot as plt


def plot_histogram(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at '{image_path}'")
        return

    # Convert to grayscale
    print("Converting image to grayscale...")
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the grey image
    # plt.imshow(grayscale_image, cmap="gray")

    # Calculate the histogram
    print("Calculating histogram...")
    hist = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.plot(hist, color="black")
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    # logscale
    plt.yscale("log")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the histogram of a grayscale image."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image.")

    args = parser.parse_args()
    plot_histogram(args.image_path)
