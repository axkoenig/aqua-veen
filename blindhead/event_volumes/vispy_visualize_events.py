"""
Minimally working realtime 3D event visualization example using VisPy.

Usage:
    python vispy_visualize_events.py --event_path path/to/your_events_file.hdf5 
                                      [--window_size 100000] 
                                      [--window_size_scale 0.001]
                                      [--update_interval 0.1]
                                      
Notes:
 - The window_size is specified in the same time unit as the event timestamps 
   in the hdf5 file (e.g. microseconds).
 - The window_size_scale multiplies the time difference to control the plotting scale.
"""

import argparse
import os
import pickle
import time

import h5py
import numpy as np
import vispy  # new import for forcing a backend

vispy.use("pyqt5")  # new: force the use of PyQt5 as the backend

# Updated import to include gloo for clearing the screen
from vispy import app, gloo, scene
from vispy.scene import visuals


# -------------------------
# Parse command-line arguments
# -------------------------
def parse_args():
    print("Parsing command line arguments...")
    parser = argparse.ArgumentParser(
        description="Realtime 3D visualization using VisPy."
    )
    parser.add_argument(
        "--event_path", type=str, required=True, help="Path to hdf5 event file."
    )
    parser.add_argument(
        "--window_size",
        type=float,
        default=100000,
        help="Time window size (in same unit as timestamps, e.g., microseconds) to display.",
    )
    parser.add_argument(
        "--window_size_scale",
        type=float,
        default=0.001,
        help="Scaling factor for the time axis.",
    )
    parser.add_argument(
        "--update_interval",
        type=float,
        default=0.1,
        help="Update interval (in seconds) for the visualization.",
    )
    return parser.parse_args()


# -------------------------
# Load events from the hdf5 file
# -------------------------
def load_events(event_path):
    print(f"Loading events from: {event_path}")
    # Create a cache file name by appending '.pkl' to the HDF5 file name.
    cache_file = event_path + ".pkl"

    if os.path.exists(cache_file):
        hdf5_mtime = os.path.getmtime(event_path)
        cache_mtime = os.path.getmtime(cache_file)
        if cache_mtime >= hdf5_mtime:
            print(f"Found up-to-date cache file at: {cache_file}")
            with open(cache_file, "rb") as f:
                events = pickle.load(f)
            print(f"Loaded {events.shape[0]} events from cache.")
            print(f"Events array size: {events.nbytes / (1024 * 1024):.2f} MB")
            return events
        else:
            print("Cache file exists but is outdated. Reloading from HDF5 file.")
    else:
        print("No cache file found. Reading from HDF5 file.")

    with h5py.File(event_path, "r") as f:
        xs = f["CD/events/x"][:]
        ys = f["CD/events/y"][:]
        ts = f["CD/events/t"][:]
        ps = f["CD/events/p"][:]
    events = np.column_stack((xs, ys, ts, ps))
    print(f"Loaded {events.shape[0]} events from {event_path}")
    print(f"Events array size: {events.nbytes / (1024 * 1024):.2f} MB")

    print(f"Caching events to {cache_file}...")
    with open(cache_file, "wb") as f:
        pickle.dump(events, f)
    print("Events cached successfully.")

    return events


# -------------------------
# Main visualization class
# -------------------------
class RealTimeEventVisualizer(scene.SceneCanvas):
    def __init__(self, events, window_size, window_size_scale, update_interval):
        print("Initializing RealTimeEventVisualizer...")
        # Store parameters first; note that we rename 'events' -> 'event_data'
        self.window_size = window_size
        self.window_size_scale = window_size_scale
        self.update_interval = update_interval
        self.event_data = events
        self.num_events = events.shape[0]
        self.start_time = self.event_data[:, 2].min()
        self.end_time = self.event_data[:, 2].max()
        self.sim_time = self.start_time
        # Add attribute to manage pause state.
        self.paused = False

        # Initialize the parent class
        super(RealTimeEventVisualizer, self).__init__(
            keys="interactive",
            size=(800, 600),
            title="Realtime 3D Event Visualization",
            show=False,  # We'll show it after full initialization
        )

        # Unfreeze the canvas to allow attribute modifications
        self.unfreeze()

        # Create grid layout and view
        grid = self.central_widget.add_grid()
        self.view = grid.add_view(row=0, col=0)
        self.view.camera = scene.TurntableCamera(elevation=30, azimuth=30)
        self.view.camera.fov = 45
        self.view.camera.distance = 500

        # Create the scatter visual
        self.scatter = visuals.Markers(parent=self.view.scene)

        # Timer for updating the visualization
        self._timer = app.Timer(self.update_interval, connect=self.on_timer, start=True)

        print(f"Simulation will run from {self.start_time} to {self.end_time}")

        # Now show the canvas
        self.show()

    def on_draw(self, event):
        """Handle draw event - clear the buffers and let SceneCanvas draw the scene."""
        gloo.clear(color="black", depth=True)
        # Let the SceneCanvas take care of rendering (calling the parent's on_draw)
        super().on_draw(event)

    def on_timer(self, event):
        """Timer callback: update simulation time and scatter data."""
        # Only update if not paused.
        if self.paused:
            return

        print(f"Timer triggered. Current simulation time: {self.sim_time}")
        # Advance simulation time; here we add update_interval in microsecond units.
        self.sim_time += (
            self.update_interval * 1e6
        )  # assuming timestamps in microseconds

        # Loop simulation if exceeding the end time.
        if self.sim_time > self.end_time:
            print("Simulation time exceeded end time. Looping back to start.")
            self.sim_time = self.start_time

        # Compute window start time.
        window_start = self.sim_time - self.window_size
        print(f"Selecting events in the time window: {window_start} to {self.sim_time}")

        # Now, select events based on the renamed attribute self.event_data.
        idx = np.where(
            (self.event_data[:, 2] >= window_start)
            & (self.event_data[:, 2] <= self.sim_time)
        )[0]
        if idx.size == 0:
            print("No events found in the current time window.")
            return  # No events to display.

        selected_events = self.event_data[idx]

        # Print the number of events currently visualized.
        print(f"Number of events visualized: {selected_events.shape[0]}")

        x_coords = selected_events[:, 0]
        y_coords = selected_events[:, 1]
        # Scale the time axis using window_size_scale after normalizing to window_start.
        t_scaled = (selected_events[:, 2] - window_start) * self.window_size_scale

        # Combine the coordinates into an array of positions (x, y, z)
        pos = np.column_stack((x_coords, y_coords, t_scaled))

        # Set colors based on polarity (we use red for p == 1, blue otherwise)
        colors = np.empty((selected_events.shape[0], 4), dtype=np.float32)
        for i, p in enumerate(selected_events[:, 3].astype(int)):
            if p == 1:
                colors[i] = (1, 0, 0, 1)  # Red
            else:
                colors[i] = (0, 0, 1, 1)  # Blue

        # Update scatter data.
        self.scatter.set_data(pos, face_color=colors, size=5)
        # Process the events to schedule a redraw
        app.process_events()
        print("Scatter data updated.")

    def on_key_press(self, event):
        """Handle key press events."""
        if event.key == "p":
            self.paused = not self.paused
            if self.paused:
                self._timer.stop()
                print("Visualization paused. Press 'p' to resume.")
            else:
                self._timer.start()
                print("Visualization resumed.")
            event.handled = True
        else:
            # Call base implementation to handle other keys
            return super().on_key_press(event)


# -------------------------
# The main entry point.
# -------------------------
def main():
    args = parse_args()
    print(f"Arguments parsed: {args}")
    events = load_events(args.event_path)
    # Create a RealTimeEventVisualizer instance (which creates a VisPy canvas)
    visualizer = RealTimeEventVisualizer(
        events=events,
        window_size=args.window_size,
        window_size_scale=args.window_size_scale,
        update_interval=args.update_interval,
    )
    print("Starting VisPy application...")
    # Run the app.
    app.run()


if __name__ == "__main__":
    main()
