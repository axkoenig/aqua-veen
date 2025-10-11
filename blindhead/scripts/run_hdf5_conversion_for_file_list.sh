#!/bin/bash
#
# This script takes in an input directory as an argument.
# It then recursively searches the subdirectories for files that match
# any of the sequences defined in the file list below, with names ending in ".raw".
# When a match is found, the conversion Python script is invoked.
#
# Usage:
#   ./run_hdf5_conversion_for_file_list.sh <input_dir>
#

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_dir>"
    exit 1
fi

input_dir="$1"

# List of sequences (file basenames without extension) to process.
sequences=(
    "video_loop_lenovo_direct_v3"
    "recording_2025-01-10_17-03-17"
    "recording_2025-01-11_15-09-13"
    "recording_2025-01-11_15-34-21"  # duplicate entry; will be processed twice unless manually filtered
    "recording_2025-01-11_19-08-18"
    "recording_2025-01-11_17-25-41"
    "recording_2025-01-11_21-24-08"
    "recording_2025-01-11_19-30-36"
    "recording_2025-01-12_15-48-06"
    "ventilator_v3_inverted"
)

# Absolute path to the Python conversion script.
conversion_script="/Volumes/T7/event_data/scripts/events_for_art/scripts/convert_raw_to_hdf5_wo_plugin.py"

for seq in "${sequences[@]}"; do
    echo "Searching for file matching \"${seq}.raw\" in ${input_dir}..."
    
    # Find the first matching raw file recursively (adjust the find command if you need a different matching strategy).
    raw_file=$(find "$input_dir" -type f -name "${seq}.raw" | head -n 1)
    
    if [ -z "$raw_file" ]; then
        echo "WARNING: No raw file found for sequence: ${seq}"
        continue
    fi

    # Get the directory that contains the raw file.
    raw_dir=$(dirname "$raw_file")
    echo "Found raw file: ${raw_file}"
    echo "Invoking conversion on directory: ${raw_dir} with sequence: ${seq}"
    
    # Run the conversion Python script. It expects the first argument to be the data root,
    # and if --sequence is provided then it converts only the file:
    #    data_root/${sequence}.raw
    python3 "$conversion_script" "$raw_dir" --sequence "$seq"
done

echo "Conversion processing complete."

