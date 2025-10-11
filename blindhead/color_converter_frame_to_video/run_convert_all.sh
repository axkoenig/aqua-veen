#!/bin/bash
set -euo pipefail

# Trap SIGINT (Control-C) to exit gracefully without processing further files.
trap 'echo -e "\033[35mInterrupted. Deleting partially created output file: $OUTPUT_FILE\033[0m" >&2; rm -f "$OUTPUT_FILE"; exit 1' SIGINT

# ANSI escape codes for colors:
MAGENTA='\033[35m'
NC='\033[0m'  # No Color / Reset to default.

# This script recursively converts all AVI files ending with "slomo_factor_3pt33"
# in a directory into a 4K color-converted MP4 version.
# If the "--delete" flag is provided, it will also remove the original files after successful conversion.
#
# Usage:
#   ./run_convert_all.sh [--delete] <directory_to_search>
#
# When using the --delete flag, you'll be shown a warning message that must be confirmed
# by pressing Enter before original files are deleted.

usage() {
    echo -e "${MAGENTA}Usage: $0 [--delete] <directory_to_search>${NC}"
    exit 1
}

# Parse arguments
DELETE_ORIGINAL=false
if [ "$#" -eq 0 ]; then
    usage
fi

if [ "$1" == "--delete" ]; then
    DELETE_ORIGINAL=true
    shift
fi

if [ "$#" -ne 1 ]; then
    usage
fi

INPUT_DIR="$1"
OUTPUT_FILE=""  # Initialize OUTPUT_FILE to an empty string

# If deletion option is enabled, warn the user and require confirmation.
if [ "$DELETE_ORIGINAL" = true ]; then
    echo -e "${MAGENTA}WARNING: You have chosen to delete original files after conversion.${NC}"
    echo -e "${MAGENTA}This action will permanently remove the original files.${NC}"
    read -rp "$(echo -e ${MAGENTA}Press [Enter] to confirm or Ctrl+C to abort...${NC})"
fi

# Create a log file with current timestamp.
TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
LOGFILE="batch_convert_${TIMESTAMP}.log"
echo "Batch conversion started at $(date)" > "$LOGFILE"
echo "Processing directory: $INPUT_DIR" >> "$LOGFILE"

# Path to the video converter binary. Update this if needed.
CONVERTER="build/video_color_converter"
if [ ! -x "$CONVERTER" ]; then
    echo -e "${MAGENTA}Converter binary not found or not executable at $CONVERTER${NC}" | tee -a "$LOGFILE"
    exit 1
fi

# Count total number of matching files.
TOTAL=$(find "$INPUT_DIR" -type f -iname '*slomo_factor_3pt33.avi' | wc -l)
REMAINING=${TOTAL}
echo -e "${MAGENTA}Total files to process: $TOTAL${NC}" | tee -a "$LOGFILE"

# Process each file. Using process substitution to avoid subshell issues.
while IFS= read -r file; do
    # Skip files that start with ._
    if [[ "$file" == *"._"* ]]; then
        continue
    fi

    echo -e "${MAGENTA}($REMAINING left) Processing file: $file${NC}" | tee -a "$LOGFILE"
    base="${file%.*}"
    output="${base}_color_converted.mp4"
    OUTPUT_FILE="$output"  # Update the output file variable

    # Check if the output file already exists.
    if [ -f "$output" ]; then
        echo -e "${MAGENTA}Output file already exists. Skipping conversion for $file.${NC}" | tee -a "$LOGFILE"
        # Still perform deletion if required.
        if [ "$DELETE_ORIGINAL" = true ]; then
            echo -e "${MAGENTA}Deleting original file: $file${NC}" | tee -a "$LOGFILE"
            rm -v "$file" >> "$LOGFILE" 2>&1
        fi
    else
        # Run the converter with the --4k flag.
        "$CONVERTER" "$file" "$output" >> "$LOGFILE" 2>&1
        if [ "$?" -eq 0 ]; then
            echo -e "${MAGENTA}Successfully converted $file to $output${NC}" | tee -a "$LOGFILE"
            if [ "$DELETE_ORIGINAL" = true ]; then
                echo -e "${MAGENTA}Deleting original file: $file${NC}" | tee -a "$LOGFILE"
                rm -v "$file" >> "$LOGFILE" 2>&1
            fi
        else
            echo -e "${MAGENTA}Error converting $file${NC}" | tee -a "$LOGFILE"
        fi
    fi
    REMAINING=$((REMAINING - 1))
done < <(find "$INPUT_DIR" -type f -iname '*slomo_factor_3pt33.avi')

echo -e "${MAGENTA}Batch conversion completed at $(date)${NC}" | tee -a "$LOGFILE"