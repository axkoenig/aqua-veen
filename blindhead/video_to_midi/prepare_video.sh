#!/bin/bash

# Video Processing Script
# This script:
# 1. Downsamples a calibration video to 1080p at 25fps
# 2. Concatenates it with a main video
# 3. Adds an audio track with a specified delay

# Display usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -c, --calibration PATH   Path to calibration video file (4K)"
    echo "  -v, --video PATH         Path to main video file (1080p)"
    echo "  -a, --audio PATH         Path to audio file (WAV)"
    echo "  -o, --offset SECONDS     Audio delay in seconds (can use decimals, e.g. 1.529)"
    echo "  -O, --output PATH        Path for output video (default: processed_video.mp4)"
    echo "  -h, --help               Display this help"
    echo
    echo "Example:"
    echo "  $0 -c calibration_4k.mp4 -v main_video.mp4 -a audio.wav -o 1.529 -O final_video.mp4"
    exit 1
}

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed or not in PATH"
    exit 1
fi

# Parse command line arguments
CALIBRATION_PATH=""
VIDEO_PATH=""
AUDIO_PATH=""
AUDIO_OFFSET=""
OUTPUT_PATH="processed_video.mp4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--calibration)
            CALIBRATION_PATH="$2"
            shift 2
            ;;
        -v|--video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        -a|--audio)
            AUDIO_PATH="$2"
            shift 2
            ;;
        -o|--offset)
            AUDIO_OFFSET="$2"
            shift 2
            ;;
        -O|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Validate required parameters
if [[ -z "$CALIBRATION_PATH" || -z "$VIDEO_PATH" || -z "$AUDIO_PATH" || -z "$AUDIO_OFFSET" ]]; then
    echo "Error: Missing required parameters"
    show_usage
fi

# Check if input files exist
for file in "$CALIBRATION_PATH" "$VIDEO_PATH" "$AUDIO_PATH"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: File not found: $file"
        exit 1
    fi
done

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
if [[ ! "$TEMP_DIR" || ! -d "$TEMP_DIR" ]]; then
    echo "Error: Could not create temporary directory"
    exit 1
fi

# Set up cleanup on exit
cleanup() {
    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Define temporary file paths
CALIBRATION_1080P="$TEMP_DIR/calibration_1080p.mp4"
CALIBRATION_TS="$TEMP_DIR/calibration.ts"
VIDEO_TS="$TEMP_DIR/video.ts"
CONCAT_VIDEO="$TEMP_DIR/concat_video.mp4"
CONCAT_LIST="$TEMP_DIR/concat_list.txt"

echo "=== Starting Video Processing ==="
echo "Step 1/4: Downsampling calibration video to 1080p at 25fps..."
ffmpeg -y -i "$CALIBRATION_PATH" -vf "scale=1080:1920" -r 25 -c:v libx264 -preset medium -crf 23 "$CALIBRATION_1080P" || {
    echo "Error during calibration video downsampling"
    exit 1
}

echo "Step 2/4: Converting videos to transport stream format..."
ffmpeg -y -i "$CALIBRATION_1080P" -c copy -bsf:v h264_mp4toannexb -f mpegts "$CALIBRATION_TS" || {
    echo "Error converting calibration video to TS format"
    exit 1
}

ffmpeg -y -i "$VIDEO_PATH" -c copy -bsf:v h264_mp4toannexb -f mpegts "$VIDEO_TS" || {
    echo "Error converting main video to TS format"
    exit 1
}

echo "Step 3/4: Concatenating videos..."
ffmpeg -y -i "concat:$CALIBRATION_TS|$VIDEO_TS" -c copy -bsf:a aac_adtstoasc "$CONCAT_VIDEO" || {
    echo "Error during video concatenation"
    exit 1
}

echo "Step 4/4: Adding audio with ${AUDIO_OFFSET}s delay..."
# Convert milliseconds for ffmpeg (multiply by 1000)
OFFSET_MS=$(echo "$AUDIO_OFFSET * 1000" | bc)
OFFSET_MS=${OFFSET_MS%.*}  # Remove decimal part

ffmpeg -y -i "$CONCAT_VIDEO" -i "$AUDIO_PATH" -filter_complex "[1:a]adelay=${OFFSET_MS}|${OFFSET_MS}[adelayed];[adelayed]apad" -c:v copy -c:a aac -shortest "$OUTPUT_PATH" || {
    echo "Error adding audio to video"
    exit 1
}

echo "=== Processing Complete ==="
echo "Final video saved to: $OUTPUT_PATH"

# Display output video information
echo "=== Output Video Information ==="
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration -of default=noprint_wrappers=1 "$OUTPUT_PATH"
ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,channels,sample_rate -of default=noprint_wrappers=1 "$OUTPUT_PATH"

echo "=== Example command to verify the results ==="
echo "ffplay \"$OUTPUT_PATH\"" 