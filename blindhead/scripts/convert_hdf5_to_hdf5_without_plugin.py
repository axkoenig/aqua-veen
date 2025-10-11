import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def h5py_loader(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        if len(f["CD"]["events"]["t"]) > 2147483647:
            print("Warning: Data size exceeds int32 limit. Please check data type.")

        total_events = len(f["CD"]["events"]["t"])
        data = {"CD": {}, "TRIGGER": {}}

        with tqdm(total=4, desc="Loading CD events") as pbar:
            for key in ["x", "y", "t", "p"]:
                dtype = (
                    np.int16
                    if key in ["x", "y"]
                    else (np.int32 if key == "t" else bool)
                )
                data["CD"][key] = np.array(f["CD"]["events"][key], dtype=dtype)
                pbar.update(1)

        with tqdm(total=2, desc="Loading Trigger events") as pbar:
            for key in ["t", "p"]:
                data["TRIGGER"][key] = np.array(f["EXT_TRIGGER"]["events"][key])
                pbar.update(1)

        return data


def h5py_saver(data: dict, path: Path):
    with h5py.File(path, "w") as f:
        with tqdm(total=6, desc="Saving events") as pbar:
            cd_group = f.create_group("CD/events")
            compression_opts = {
                "compression": "gzip",
                "compression_opts": 9,
                "shuffle": True,
                "chunks": True,
            }

            for key, dtype in [
                ("x", np.int16),
                ("y", np.int16),
                ("t", np.int32),
                ("p", bool),
            ]:
                cd_group.create_dataset(
                    key, data=data["CD"][key], dtype=dtype, **compression_opts
                )
                pbar.update(1)

            trigger_group = f.create_group("EXT_TRIGGER/events")
            for key in ["t", "p"]:
                trigger_group.create_dataset(
                    key,
                    data=data["TRIGGER"][key],
                    dtype=data["TRIGGER"][key].dtype,
                    **compression_opts,
                )
                pbar.update(1)
    print(f"Data saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert HDF5 event files")
    parser.add_argument("input_path", type=Path, help="Path to input HDF5 file")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = input_path.parent / f"{input_path.stem}_no_plugin{input_path.suffix}"

    events = h5py_loader(input_path)
    h5py_saver(events, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
