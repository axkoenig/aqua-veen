import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import streamlit as st
from PIL import Image


@st.cache_data
def load_event_data(event_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load event data from an HDF5 file and cache it."""
    with h5py.File(event_path, "r") as f:
        event_timestamps = f["CD/events/t"][:]
        events_data = {
            "x": f["CD/events/x"][:],
            "y": f["CD/events/y"][:],
            "t": f["CD/events/t"][:],
            "p": f["CD/events/p"][:],
        }
    return event_timestamps, events_data


def make_overlay_image(
    events: np.ndarray,
    height: int,
    width: int,
    color_background: str,
    color_positive: str,
    color_negative: str,
    pixel_size: int,
) -> np.ndarray:
    """Create an overlay image based on events."""
    background_color = tuple(int(color_background[i : i + 2], 16) for i in (1, 3, 5))
    positive_color = tuple(int(color_positive[i : i + 2], 16) for i in (1, 3, 5))
    negative_color = tuple(int(color_negative[i : i + 2], 16) for i in (1, 3, 5))

    image = np.full((height, width, 3), background_color, dtype=np.uint8)

    for i in range(0, len(events)):
        x, y, p = events[i][0], events[i][1], events[i][3]
        block_x = x // pixel_size
        block_y = y // pixel_size

        if block_x < width // pixel_size and block_y < height // pixel_size:
            if p == 0:
                image[
                    block_y * pixel_size : (block_y + 1) * pixel_size,
                    block_x * pixel_size : (block_x + 1) * pixel_size,
                ] = negative_color
            else:
                image[
                    block_y * pixel_size : (block_y + 1) * pixel_size,
                    block_x * pixel_size : (block_x + 1) * pixel_size,
                ] = positive_color

    return image


def scale_events(
    events_data: Dict[str, np.ndarray],
    original_height: int,
    original_width: int,
    target_height: int,
    target_width: int,
) -> Dict[str, np.ndarray]:
    """Scale event data to fit the target resolution."""
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    events_data["x"] = (events_data["x"] * scale_x).astype(int)
    events_data["y"] = (events_data["y"] * scale_y).astype(int)

    return events_data


def process_single_overlay(
    events_data: Dict[str, np.ndarray],
    event_timestamps: np.ndarray,
    timestamp: int,
    height: int,
    width: int,
    num_events: int,
    color_background: str,
    color_positive: str,
    color_negative: str,
    pixel_size: int,
) -> np.ndarray:
    """Process and generate a single overlay image."""
    event_idx = np.searchsorted(event_timestamps, timestamp)

    start_idx = max(0, event_idx - num_events // 2)
    end_idx = min(len(event_timestamps), event_idx + num_events // 2)

    if start_idx > end_idx:
        st.error("No events found in the selected time range.")
        return np.zeros((height, width, 3), dtype=np.uint8)
    else:
        events = np.zeros((end_idx - start_idx, 4), dtype=int)
        events[:, 0] = events_data["x"][start_idx:end_idx]
        events[:, 1] = events_data["y"][start_idx:end_idx]
        events[:, 2] = events_data["t"][start_idx:end_idx]
        events[:, 3] = events_data["p"][start_idx:end_idx]

        events = events[events[:, 0] < width]
        events = events[events[:, 1] < height]
        overlay = make_overlay_image(
            events,
            height,
            width,
            color_background,
            color_positive,
            color_negative,
            pixel_size,
        )

    return overlay


def save_config(config: Dict[str, any], filename: str) -> None:
    """Save configuration as a JSON file."""
    with open(filename, "w") as f:
        json.dump(config, f, indent=4)


def load_config(filename: str) -> Dict[str, any]:
    """Load configuration from a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def get_default_config() -> Dict[str, any]:
    """Return the default configuration."""
    return {
        "file_index": 0,
        "resolution_index": 3,
        "num_events": 100000,
        "color_background": "#000000",
        "color_positive": "#1111aa",
        "color_negative": "#FF0000",
        "rotation_option": "Rotate Left",
        "pixel_size": 7,
        "tag": "overlay",
    }


def main(
    data_dir: str = "/Users/koenig/Documents/personal/art projects/colab justin/blindhead/proc/data",
    output_dir: str = "/Users/koenig/Documents/personal/art projects/colab justin/blindhead/proc/events_for_art/visualizer/output",
) -> None:
    """Main function to run the Streamlit app."""
    with st.sidebar:
        st.title("Event Visualizer")
        st.header("Settings")

        if not os.path.exists(data_dir):
            st.error("Invalid directory path")
            return

        availabel_configs = [f for f in os.listdir(output_dir) if f.endswith(".json")]

        if availabel_configs:
            availabel_configs.sort(reverse=True)
            config_file = st.selectbox(
                "Select Configuration",
                availabel_configs,
                index=st.session_state.get("config_index", 0),
            )
            config = load_config(Path(output_dir) / config_file)
            st.session_state.update(config)
        else:
            config = get_default_config()

        available_files = [f for f in os.listdir(data_dir) if f.endswith(".hdf5")]
        file_name = st.selectbox(
            "Select File",
            available_files,
            index=st.session_state.get("file_index", 0),
        )
        event_path = Path(data_dir) / file_name

        resolution = st.selectbox(
            "Output Resolution",
            ["480p", "HD", "4k", "8k"],
            index=st.session_state.get("resolution_index", 1),
        )
        resolution_map = {
            "480p": (480, 854),
            "HD": (720, 1280),
            "4k": (2160, 3840),
            "8k": (4320, 7680),
        }
        height, width = resolution_map[resolution]

        num_events = st.slider("Number of Events", 1000, 100000, 60000, 1000)

        col1, col2, col3 = st.columns(3)
        st.write("Colors")
        with col1:
            color_background = st.color_picker("Background", "#FFFFFF")
        with col2:
            color_positive = st.color_picker("Pos. Event", "#0000FF")
        with col3:
            color_negative = st.color_picker("Neg. Event", "#FF0000")

        rotation_option = st.selectbox(
            "Rotate Image",
            ["Rotate Left", "Rotate Right", "Rotate 180°", "No Rotation"],
            index=0,
        )

        pixel_size = st.slider("Pixel Size", 1, 20, 1)

        tag = st.text_input("Tag Output File", "overlay")

        event_timestamps, events_data = load_event_data(event_path)

        min_time = event_timestamps.min() / 1_000_000
        max_time = event_timestamps.max() / 1_000_000

        timestamp = st.slider(
            "Select Timestamp (seconds)", min_time, max_time, min_time
        )

    with st.spinner("Generating overlay..."):
        original_height, original_width = 720, 1280

        if resolution != "HD":
            with st.spinner("Scaling event data..."):
                events_data = scale_events(
                    events_data, original_height, original_width, height, width
                )

        overlay = process_single_overlay(
            events_data,
            event_timestamps,
            int(timestamp * 1_000_000),
            height,
            width,
            num_events,
            color_background,
            color_positive,
            color_negative,
            pixel_size,
        )

        with st.spinner("Processing overlay..."):
            overlay_image = Image.fromarray(overlay)

            if rotation_option == "Rotate Left":
                overlay_image = overlay_image.rotate(90, expand=True)
            elif rotation_option == "Rotate Right":
                overlay_image = overlay_image.rotate(-90, expand=True)
            elif rotation_option == "Rotate 180°":
                overlay_image = overlay_image.rotate(180, expand=True)

            st.image(overlay_image)

    if st.button("Save Overlay"):
        current_date = datetime.now().strftime("%Y-%m-%d")
        output_filename = f"{tag}_{current_date}_{timestamp}.png"
        output_path = Path(output_dir) / output_filename
        output_path.parent.mkdir(exist_ok=True)
        overlay_image.save(output_path)

        config = {
            "file_index": available_files.index(file_name),
            "resolution_index": ["480p", "HD", "4k", "8k"].index(resolution),
            "num_events": num_events,
            "color_background": color_background,
            "color_positive": color_positive,
            "color_negative": color_negative,
            "rotation_option": rotation_option,
            "pixel_size": pixel_size,
            "tag": tag,
        }
        config_filename = output_path.with_suffix(".json")
        save_config(config, config_filename)

        st.success(f"Overlay saved to: {output_path}")
        st.success(f"Configuration saved to: {config_filename}")


if __name__ == "__main__":
    main()
