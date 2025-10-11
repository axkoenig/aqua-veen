#!/bin/bash
set -e

# Define the build directory.
BUILD_DIR="build"

# Create the build directory if it doesn't exist.
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir "$BUILD_DIR"
fi

# Compile the source file with FFmpeg flags and specify the min. macOS version.
echo "Compiling main.cpp..."
g++ -std=c++11 -mmacosx-version-min=14.0 main.cpp -o "$BUILD_DIR/video_color_converter" $(pkg-config --cflags --libs libavformat libavcodec libswscale libavutil)
echo "Compilation complete."

# Check if the input argument (video file) is provided.
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_video>"
    exit 1
fi

INPUT_VIDEO="$1"

# Create the output filename by inserting "_color_converted" before the file extension.
base="${INPUT_VIDEO%.*}"
extension="mp4"
OUTPUT_VIDEO="${base}_color_converted_4k.${extension}"

echo "Processing video: $INPUT_VIDEO"
echo "Output will be saved to: $OUTPUT_VIDEO"

# Run the video color converter executable.
"$BUILD_DIR/video_color_converter" "$INPUT_VIDEO" "$OUTPUT_VIDEO" --4k