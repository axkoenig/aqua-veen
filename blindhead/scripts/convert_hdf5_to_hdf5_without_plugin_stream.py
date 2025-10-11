import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def process_in_chunks(input_path: Path, output_path: Path, chunk_size: int = 1000000):
    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        # Get total number of events
        total_events = len(fin["CD"]["events"])
        if total_events > 2147483647:
            print("Warning: Data size exceeds int32 limit. Please check data type.")

        # Setup output file structure with compression
        compression_opts = {
            "compression": "gzip",
            "compression_opts": 9,
            "shuffle": True,
            "chunks": True,
        }

        # Create the output datasets
        cd_group = fout.create_group("CD/events")
        for key, dtype in [
            ("x", np.int16),
            ("y", np.int16),
            ("t", np.int32),
            ("p", bool),
        ]:
            # This creates the empty dataset that we'll fill in chunks
            cd_group.create_dataset(
                key, shape=(total_events,), dtype=dtype, **compression_opts
            )

        # Process CD events in chunks
        with tqdm(total=total_events, desc="Processing CD events") as pbar:
            # Process 1M events at a time
            for i in range(0, total_events, chunk_size):
                # Read chunk from input file
                chunk = {
                    "x": fin["CD"]["events"][i : i + chunk_size]["x"],
                    "y": fin["CD"]["events"][i : i + chunk_size]["y"],
                    "t": fin["CD"]["events"][i : i + chunk_size]["t"],
                    "p": fin["CD"]["events"][i : i + chunk_size]["p"],
                }

                # Write chunk to output file
                for key in ["x", "y", "t", "p"]:
                    cd_group[key][i : i + chunk_size] = chunk[key]

                pbar.update(len(chunk["x"]))
                del chunk  # Free memory

        # Process trigger events (small enough to do at once)
        trigger_group = fout.create_group("EXT_TRIGGER/events")
        print("Processing Trigger events")
        for key in ["t", "p"]:
            trigger_group.create_dataset(
                key, data=fin["EXT_TRIGGER"]["events"][key][:], **compression_opts
            )

    print(f"Data saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert HDF5 event files")
    parser.add_argument("input_path", type=Path, help="Path to input HDF5 file")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = input_path.parent / f"{input_path.stem}_no_plugin{input_path.suffix}"

    process_in_chunks(input_path, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
