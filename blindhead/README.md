# Events for Art

## Minimal Workflow from Recording to Visualization

1. Record using `metavision_studio`, saving to a .raw file
2. Convert to Prophesee's hdf5 format using `metavision_file_to_hdf5 -i <raw-file-path>`
3. Convert to generic hdf5 format with

        python scripts/convert_hdf5_to_hdf5_without_plugin.py <path-to-hdf5>

4. Now on any other machine only needing h5py as dependency we can use

        python scripts/visualize_events.py <path-to-hdf5-no-plugin>