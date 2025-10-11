import argparse
from pathlib import Path
import h5py
import numpy as np
from metavision_core.event_io import EventsIterator
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_root',
        type=Path
    )
    parser.add_argument(
        '--output_root',
        type=Path,
        default=None
    )
    parser.add_argument(
        '--sequence',
        type=str,
        default=''
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=300000
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None
    )
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = args.data_root
    if args.num_workers is None:
        args.num_workers = mp.cpu_count()
    return args

def convert_raw_to_hdf5(raw_path: Path, hdf_path: Path, chunk_size: int = 300000):
    print(f'Converting {raw_path} to {hdf_path}')
    
    events_iterator = EventsIterator(
        input_path=str(raw_path),
        mode="n_events",
        n_events=chunk_size
    )
    
    with h5py.File(hdf_path, 'w') as f:
        cd_group = f.create_group("CD/events")
        compression_opts = {
            'compression': 'gzip',
            'compression_opts': 4,
            'shuffle': True,
            'chunks': True
        }
        
        datasets = {}
        first_chunk = True
        
        for events in tqdm(events_iterator, desc="Processing events", position=mp.current_process()._identity[0]):
            if first_chunk:
                for key in ['x', 'y', 't', 'p']:
                    dtype = np.int16 if key in ['x', 'y'] else (np.int32 if key == 't' else bool)
                    datasets[key] = cd_group.create_dataset(
                        key,
                        data=events[key],
                        maxshape=(None,),
                        dtype=dtype,
                        **compression_opts
                    )
                first_chunk = False
            else:
                for key in ['x', 'y', 't', 'p']:
                    n_old = datasets[key].shape[0]
                    n_new = n_old + len(events[key])
                    datasets[key].resize(n_new, axis=0)
                    datasets[key][n_old:] = events[key]
        
        trigger_events = events_iterator.get_ext_trigger_events()
        if len(trigger_events) > 0:
            trigger_group = f.create_group("EXT_TRIGGER/events")
            for key in ['t', 'p']:
                trigger_group.create_dataset(
                    key,
                    data=trigger_events[key],
                    **compression_opts
                )
    return True

def process_sequence(args):
    raw_path, output_dir, chunk_size = args
    sequence = raw_path.stem
    hdf_path = output_dir / f'{sequence}.h5'
    
    if hdf_path.exists():
        print(f'Output file already exists, skipping: {hdf_path}')
        return None
    
    try:
        success = convert_raw_to_hdf5(raw_path, hdf_path, chunk_size)
        if not success:
            if hdf_path.exists():
                hdf_path.unlink()
    except Exception as e:
        print(f'Error processing {sequence}: {str(e)}')
        if hdf_path.exists():
            hdf_path.unlink()
        return None
    return sequence

def main():
    args = parse_args()
    data_root = args.data_root.expanduser()
    output_root = args.output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    
    if args.sequence:
        raw_files = [data_root / f"{args.sequence}.raw"]
        if not raw_files[0].exists():
            print(f"RAW file not found: {raw_files[0]}")
            return
    else:
        raw_files = sorted(data_root.glob('*.raw'))
        print(f'Found RAW files: {[f.name for f in raw_files]}')
    
    process_args = [(raw_path, output_root, args.chunk_size) for raw_path in raw_files]
    
    with Pool(args.num_workers) as pool:
        results = pool.map(process_sequence, process_args)
    
    completed = [r for r in results if r is not None]
    print(f'Successfully converted {len(completed)} sequences')
    print('Conversion completed.')

if __name__ == '__main__':
    main()