import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def make_overlay_image(events, height=260, width=346):
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x, y, p = events[:, 0], events[:, 1], events[:, 3]
    image[y[p == 0], x[p == 0]] = (255, 0, 0)
    image[y[p == 1], x[p == 1]] = (0, 0, 255)
    return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event_path", help="Path to converted hdf5 file", type=Path, required=True
    )
    parser.add_argument("--height", type=int, default=720, help="Image height")
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument(
        "--stride", type=float, default=0.033, help="Time step in seconds"
    )
    parser.add_argument(
        "--num_events", type=int, default=60000, help="Number of events in a frame"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = args.event_path.parent / f"{args.event_path.stem}_overlays"
    output_dir.mkdir(exist_ok=True)

    with h5py.File(args.event_path, "r") as f:
        event_timestamps = f["CD/events/t"][:]  # us
        min_time = event_timestamps.min()
        max_time = event_timestamps.max()
        img_timestamps = np.arange(min_time, max_time, args.stride * 1e6)
        event_indices = np.searchsorted(event_timestamps, img_timestamps)

        for img_idx, event_idx in tqdm(
            enumerate(event_indices),
            total=len(event_indices),
            desc="Generating visualizations",
        ):
            start_idx = max(0, event_idx - args.num_events // 2)
            end_idx = min(len(event_timestamps), event_idx + args.num_events // 2)

            if end_idx - start_idx > 0:
                events = np.zeros((end_idx - start_idx, 4), dtype=int)
                events[:, 0] = f["CD/events/x"][start_idx:end_idx]
                events[:, 1] = f["CD/events/y"][start_idx:end_idx]
                events[:, 2] = f["CD/events/t"][start_idx:end_idx]
                events[:, 3] = f["CD/events/p"][start_idx:end_idx]

                events = events[events[:, 0] < args.width]
                events = events[events[:, 1] < args.height]
                overlay = make_overlay_image(events, args.height, args.width)

            else:
                overlay = np.full((args.height, args.width, 3), 255, dtype=np.uint8)

            Image.fromarray(overlay).save(
                output_dir / f"overlay_{str(img_idx).zfill(5)}.png"
            )
