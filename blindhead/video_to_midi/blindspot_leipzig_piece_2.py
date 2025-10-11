"""
Takes video file and plays Analog Four.
"""

import argparse
import hashlib
import math
import multiprocessing
import os
import pickle
import signal
import subprocess
import time
from collections import deque
import json
import socket
import tempfile
import shutil

import cv2
import matplotlib.pyplot as plt
import mido
import numpy as np
import pandas as pd
from tqdm import tqdm

# Constants
# (Note: grid-related constants have been removed because we now process
#  the entire image via region detection.)
MIDI_RANGE = 127
BRIGHTNESS_RANGE = 255
PLOT_HISTORY_SECONDS = 3
NUM_SHAPES = 4
MAX_REVERB_DELAY = 60  # Maximum value for reverb and delay sends
DEBUG = False  # Global debug flag
MIDI_CHANNEL = 0
SYNC_SEQUENCE_DURATION = 30  # Duration of sync sequence in seconds
BLACK_SCREEN_DURATION = 5  # Duration of black screen at the beginning in seconds
POST_CONVERGENCE_PAUSE = 5  # Duration of pause after circles converge in seconds

# Color mapping for channels (BGR format for OpenCV)
CHANNEL_COLORS = {
    0: (0, 0, 255),    # Red
    1: (0, 255, 0),    # Green  
    2: (255, 0, 0),    # Blue
    3: (0, 255, 255),  # Yellow
}

# Shape counters for each channel
SHAPE_COUNTERS = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
}

def debug_print(*args, **kwargs):
    """Print only when DEBUG is True"""
    if DEBUG:
        print(*args, **kwargs)


def init_midi_output():
    """Initialize MIDI output."""
    try:
        output_names = mido.get_output_names()
        print(f"MIDI Outputs: {output_names}")
        if not output_names:
            print("No MIDI output available.")
            return None

        midi_out = mido.open_output(output_names[0])
        print(f"Opened MIDI output: {output_names[0]}")
        return midi_out
    except Exception as e:
        print(f"Error initializing MIDI: {e}")
        return None


# Add global state for smoothing ambient effects
class AmbientEffectsState:
    def __init__(self):
        self.current_effect_value = MAX_REVERB_DELAY / 2  # Start in the middle
        self.smoothing_factor = 1  # Lower value = smoother transitions


# Initialize the ambient effects state
ambient_effects_state = AmbientEffectsState()


def send_midi_control_change(
    midi_out,
    mapping,
    parameter_name,
    value,
    section="Filter parameters",
    channel=MIDI_CHANNEL,
):
    """
    Sends a MIDI control change message for a filter parameter based on a CSV mapping.
    (This function is retained from the previous version even though it is not used in
    the new note-playing logic.)
    """
    row = mapping[
        (mapping["section"] == section) & (mapping["parameter_name"] == parameter_name)
    ]
    if row.empty:
        debug_print(f"Parameter '{parameter_name}' not found in the mapping.")
        return

    cc_msb = int(row.iloc[0]["cc_msb"])
    cc_min_value = int(row.iloc[0]["cc_min_value"])
    cc_max_value = int(row.iloc[0]["cc_max_value"])

    clamped_value = max(cc_min_value, min(cc_max_value, value))
    msg = mido.Message(
        "control_change", channel=channel, control=cc_msb, value=clamped_value
    )
    midi_out.send(msg)
    debug_print(
        f"Sent '{parameter_name}' message on channel {channel}: control {cc_msb}, value {clamped_value}"
    )


def send_ambient_effects(midi_out, frame, midi_mapping, channel=MIDI_CHANNEL):
    """
    Calculate average brightness of frame and inversely map to delay and reverb sends.
    Black (dark) = maximum reverb/delay, White (bright) = minimum reverb/delay

    Now applies to all channels that might be in use (0 to NUM_SHAPES-1).
    """
    if midi_out is None:
        return

    # Get access to global state
    global ambient_effects_state

    # Calculate average brightness (0-255)
    avg_brightness = np.mean(frame)

    # Inverse mapping: 255 (white) → 0, 0 (black) → MAX_REVERB_DELAY
    # Map from 0-255 to MAX_REVERB_DELAY-0
    target_effect_value = int(
        MAX_REVERB_DELAY * (1 - avg_brightness / BRIGHTNESS_RANGE)
    )

    # Apply smoothing to effect value
    alpha = ambient_effects_state.smoothing_factor
    ambient_effects_state.current_effect_value = (
        ambient_effects_state.current_effect_value * (1 - alpha)
        + target_effect_value * alpha
    )
    smoothed_effect_value = int(ambient_effects_state.current_effect_value)

    # Send MIDI messages for delay and reverb to all potential channels
    for channel in range(NUM_SHAPES):
        send_midi_control_change(
            midi_out,
            midi_mapping,
            "Amp: Delay Send",
            smoothed_effect_value,
            section="Amp",
            channel=channel,
        )

        send_midi_control_change(
            midi_out,
            midi_mapping,
            "Amp: Reverb Send",
            smoothed_effect_value,
            section="Amp",
            channel=channel,
        )

    debug_print(
        f"Average brightness: {avg_brightness:.2f}, Effect value: {smoothed_effect_value}"
    )


def draw_regions(frame):
    """
    Draw the contours for up to NUM_SHAPES largest disconnected regions above a brightness
    threshold of 40 onto a copy of the provided frame.
    """
    # Convert frame to grayscale for thresholding.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Threshold the image: pixels with brightness >=40 become white.
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        top_contours = sorted_contours[:NUM_SHAPES]
        cv2.drawContours(frame, top_contours, -1, (0, 255, 0), 2)
    return frame


class Shape:
    """
    Represents a detected bright region that plays a persistent MIDI note or chord.
    The notes are sampled from C# minor scale notes when created and continue
    playing until the shape is destroyed. The number of notes played depends on
    the shape's size relative to the frame.
    """

    # C# minor scale MIDI notes (spanning 3 octaves)
    SCALE_NOTES = [
        61,
        63,
        64,
        66,
        68,
        69,
        71,  # First octave
        73,
        75,
        76,
        78,
        80,
        81,
        83,  # Second octave
        85,
        87,
        88,
        90,
        92,
        93,
        95,  # Third octave
    ]

    def __init__(
        self,
        centroid,
        area,
        frame_dims,
        midi_out,
        midi_mapping=None,
        channel=MIDI_CHANNEL,
        shape_counter=0,
    ):
        """
        Initialize a new shape with notes from scale based on size.
        Immediately starts playing the notes.
        """
        self.shape_counter = shape_counter
        self.centroid = centroid
        self.area = area
        self.frame_width, self.frame_height = frame_dims
        self.midi_out = midi_out
        self.midi_mapping = midi_mapping
        self.channel = (
            channel  # Now set to a unique channel by the process_frame function
        )
        self.missing_frames = 0  # Counter for frames where shape is not detected
        self.max_missing_frames = 30  # Allow shape to be missing for up to 10 frames
        self.is_active = True  # Flag to track if shape is currently playing
        
        # Store contour for overlay display
        self.contour = None

        # Add filter smoothing properties
        self.current_filter_value = 64  # Start at middle value
        self.filter_smoothing_factor = 0.1  # Lower = smoother transitions

        # Determine number of notes based on area relative to frame
        self.frame_area = self.frame_width * self.frame_height
        self.normalized_area = self.area / self.frame_area
        self.num_notes = self.map_shape_area_to_num_notes()

        # Select chord notes
        self.notes = self._select_chord_notes()

        # Calculate initial velocity based on area
        self.velocity = self._calculate_velocity()

        # Start playing the notes
        self._send_note_on()
        debug_print(
            f"Created shape at {centroid} with notes {self.notes} and velocity {self.velocity} on channel {self.channel}"
        )


    def _select_chord_notes(self):
        """Select a chord of appropriate size from the scale."""
        # Validate that we have enough notes in the scale
        if len(self.SCALE_NOTES) < 7:
            print(f"Error: SCALE_NOTES must have at least 7 notes, but has {len(self.SCALE_NOTES)}")
            # Return a safe default (first note repeated)
            return [self.SCALE_NOTES[0]] * min(self.num_notes, len(self.SCALE_NOTES))
        
        # Pick a random starting note from the scale
        # For 4-note chords, we need start_idx + 6 to be valid, so max start_idx = len(SCALE_NOTES) - 7
        max_start_idx = len(self.SCALE_NOTES) - 7
        start_idx = np.random.randint(0, max_start_idx + 1)  # +1 because randint is exclusive on upper bound

        # For single note, just return it
        if self.num_notes == 1:
            return [self.SCALE_NOTES[start_idx]]

        # For 2-note chord, use root and third (skipping a scale note)
        elif self.num_notes == 2:
            return [self.SCALE_NOTES[start_idx], self.SCALE_NOTES[start_idx + 2]]

        # For 3-note chord, use root, third, and fifth (standard triad)
        elif self.num_notes == 3:
            return [self.SCALE_NOTES[start_idx], self.SCALE_NOTES[start_idx + 2], self.SCALE_NOTES[start_idx + 4]]

        # For 4-note chord, use root, third, fifth, and seventh (standard tetrad)
        else:
            return [
                self.SCALE_NOTES[start_idx],
                self.SCALE_NOTES[start_idx + 2],
                self.SCALE_NOTES[start_idx + 4],
                self.SCALE_NOTES[start_idx + 6],
            ]

    def _calculate_velocity(self):
        """Calculate MIDI velocity (30-127) based on shape's area relative to frame."""
        velocity = int(30 + self.normalized_area * (127 - 30))
        return max(30, min(127, velocity))

    def _send_note_on(self):
        """Send MIDI note-on messages for all notes in the chord."""
        if self.midi_out:
            for note in self.notes:
                self.midi_out.send(
                    mido.Message(
                        "note_on",
                        note=note,
                        velocity=self.velocity,
                        channel=self.channel,
                    )
                )

    def map_shape_area_to_num_notes(self):
        """Map shape area to number of notes."""
        # if self.normalized_area > 0.50:  # > 50% of frame
        #     return 4
        if self.normalized_area > 0.5:  # > 25% of frame
            return 3
        elif self.normalized_area > 0.25:  # > 10% of frame
            return 2
        else:
            return 1

    def update(self, centroid, area, frame, contour):
        """
        Update shape properties and adjust MIDI parameters.
        Called periodically to update filter parameters.
        """
        # Validate frame
        if frame is None or frame.size == 0:
            debug_print("Warning: Invalid frame in Shape.update(), skipping update")
            return
        
        # Check if size changed enough to warrant changing chord
        old_area = self.area
        self.centroid = centroid
        self.area = area
        self.normalized_area = self.area / self.frame_area
        self.missing_frames = 0  # Reset missing frames counter since shape was detected
        self.contour = contour  # Store contour for overlay display
        
        # Determine new chord size
        new_num_notes = self.map_shape_area_to_num_notes()

        # If chord size changed, update notes
        if new_num_notes != self.num_notes:
            # Stop current notes
            if self.midi_out:
                for note in self.notes:
                    self.midi_out.send(
                        mido.Message(
                            "note_off", note=note, velocity=0, channel=self.channel
                        )
                    )

            # Update chord
            self.num_notes = new_num_notes
            self.notes = self._select_chord_notes()
            self._send_note_on()
            debug_print(
                f"Updated shape to play notes {self.notes} on channel {self.channel}"
            )

        # Create a mask for the shape
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

        # Calculate average brightness of the shape
        mean_brightness = cv2.mean(frame, mask=mask)[0]

        # Map brightness (0-255) to filter cutoff (0-127)
        filter_cut_off_start = 40
        target_filter_cutoff = int(mean_brightness * 127 / 255) + filter_cut_off_start

        # Apply smoothing to filter value
        alpha = self.filter_smoothing_factor
        self.current_filter_value = (
            self.current_filter_value * (1 - alpha) + target_filter_cutoff * alpha
        )
        smoothed_filter_value = int(self.current_filter_value)

        # Send MIDI CC message for filter cutoff using the mapping
        if self.midi_out and self.midi_mapping is not None:
            send_midi_control_change(
                self.midi_out,
                self.midi_mapping,
                "Filter 1: Frequency",
                smoothed_filter_value,
                section="Filters",
                channel=self.channel,
            )

    def destroy(self):
        """
        Stop playing all notes and clean up.
        Called when shape is no longer detected for too long.
        """
        if self.midi_out and self.is_active:
            for note in self.notes:
                self.midi_out.send(
                    mido.Message(
                        "note_off", note=note, velocity=0, channel=self.channel
                    )
                )
            self.is_active = False
        debug_print(
            f"Destroyed shape with notes {self.notes} on channel {self.channel}"
        )


def process_frame(midi_out, frame, active_shapes, midi_mapping, max_shapes=None):
    """
    Process frame and manage shape lifecycle.
    Returns updated list of active shapes.
    
    Args:
        midi_out: MIDI output device
        frame: Frame to process
        active_shapes: List of currently active shapes
        midi_mapping: MIDI mapping configuration
        max_shapes: Maximum number of shapes to detect (defaults to NUM_SHAPES)
    """
    global SHAPE_COUNTERS
    if max_shapes is None:
        max_shapes = NUM_SHAPES
    
    height, width = frame.shape[:2]
    frame_dims = (width, height)

    # Calculate minimum area threshold
    min_area = (width / 50) * (height / 50)

    # Detect regions
    _, thresh = cv2.threshold(frame, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # First increment missing frames counter for all shapes
    for shape in active_shapes:
        shape.missing_frames += 1

    # Filter and sort contours by area, limiting to max_shapes
    valid_contours = []
    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[
            :max_shapes
        ]

    if not valid_contours:
        # No valid contours found, check if any shapes should be destroyed
        new_active_shapes = []
        for shape in active_shapes:
            if shape.missing_frames <= shape.max_missing_frames:
                new_active_shapes.append(shape)
            else:
                shape.destroy()
        return new_active_shapes

    # Calculate centroids for current frame's shapes
    current_centroids = []
    for contour in valid_contours:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            cx, cy = width // 2, height // 2
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        current_centroids.append((cx, cy, cv2.contourArea(contour), contour))

    # Match current shapes with active shapes
    new_active_shapes = []
    used_current = set()
    used_active = set()

    # Try to match existing shapes first
    for i, (cx, cy, area, contour) in enumerate(current_centroids):
        best_match = None
        best_distance = float("inf")
        best_idx = None

        for j, shape in enumerate(active_shapes):
            if j in used_active:
                continue
            distance = math.sqrt(
                (cx - shape.centroid[0]) ** 2 + (cy - shape.centroid[1]) ** 2
            )
            if distance < 100 and distance < best_distance:
                best_match = shape
                best_distance = distance
                best_idx = j

        if best_match is not None:
            # Update existing shape
            best_match.update((cx, cy), area, frame, contour)
            new_active_shapes.append(best_match)
            used_current.add(i)
            used_active.add(best_idx)

    # Check if unmatched shapes should be kept (based on missing_frames counter)
    for i, shape in enumerate(active_shapes):
        if i not in used_active:
            if shape.missing_frames <= shape.max_missing_frames:
                new_active_shapes.append(shape)
            else:
                shape.destroy()

    # Create new shapes for unmatched current shapes (if we have room)
    if len(new_active_shapes) < max_shapes:
        # Get list of channels currently in use
        used_channels = {shape.channel for shape in new_active_shapes}

        for i, (cx, cy, area, contour) in enumerate(current_centroids):
            if i not in used_current:
                # Find the first available channel
                channel = 0
                while channel < max_shapes and channel in used_channels:
                    channel += 1

                # Create new shape with the assigned channel
                SHAPE_COUNTERS[channel] += 1
                new_shape = Shape(
                    (cx, cy), area, frame_dims, midi_out, midi_mapping, channel=channel, shape_counter=SHAPE_COUNTERS[channel]
                )
                # Set the contour for the new shape
                new_shape.contour = contour
                new_active_shapes.append(new_shape)
                used_channels.add(channel)

                # Only add shapes until we hit max_shapes
                if len(new_active_shapes) >= max_shapes:
                    break

    return new_active_shapes


def send_stop_message(midi_out):
    """Send a stop message to all MIDI channels."""
    if midi_out:
        for channel in range(16):
            for note in range(128):
                midi_out.send(mido.Message("note_off", note=note, channel=channel))





class ExternalVideoPlayer:
    """
    External HD video player with persistent window using mpv.
    Uses mpv IPC control for pause/seek without restarting.
    """
    
    def __init__(self, video_path):
        """
        Initialize external player controller.
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path
        self.process = None
        self.current_position = 0
        self.is_paused = False
        self.paused_at_position = 0
        # mpv IPC fields
        self.ipc_path = None
        self._ipc_sock = None
        
    def _mpv_wait_for_ipc(self, timeout_seconds):
        """Wait until mpv creates the IPC socket."""
        start = time.time()
        while time.time() - start < timeout_seconds:
            if self.ipc_path and os.path.exists(self.ipc_path):
                try:
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    sock.settimeout(0.2)
                    sock.connect(self.ipc_path)
                    self._ipc_sock = sock
                    return True
                except Exception:
                    time.sleep(0.05)
            else:
                time.sleep(0.05)
        return False
    
    def _mpv_send(self, command_list):
        """Send a JSON IPC command to mpv."""
        if not self._ipc_sock:
            return False
        try:
            payload = json.dumps({"command": command_list}) + "\n"
            # mpv expects newline-delimited JSON
            self._ipc_sock.sendall(payload.encode('utf-8'))
            return True
        except Exception:
            return False
    
    def start(self, start_position=0):
        """Start the external player from a specific position without losing window."""
        # Create a unique IPC socket path
        try:
            tmp_dir = tempfile.gettempdir()
            self.ipc_path = os.path.join(
                tmp_dir, f"mpv-ipc-{os.getpid()}-{int(time.time())}.sock"
            )
            if os.path.exists(self.ipc_path):
                try:
                    os.unlink(self.ipc_path)
                except Exception:
                    pass
            cmd = [
                'mpv',
                '--no-terminal',
                '--force-window=yes',
                "--loop-file=inf",
                "--keep-open=yes",
                "--title=HD Video",
                "--geometry=+100+100",
                f"--input-ipc-server={self.ipc_path}",
            ]
            if start_position > 0:
                cmd.append(f"--start={start_position}")
            cmd.append(self.video_path)
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait for IPC to be available
            if not self._mpv_wait_for_ipc(5.0):
                print("Error: mpv IPC did not become ready in time")
                return False
            self.current_position = start_position
            print(
                f"Started mpv for HD video playback from {start_position:.2f}s"
            )
            return True
        except FileNotFoundError:
            print("Error: mpv not found. Please install it first.")
            print("  macOS: brew install mpv")
            print("  Linux: apt-get install mpv")
            return False
        except Exception as e:
            print(f"Error starting mpv: {e}")
            return False
    
    def seek(self, position_seconds):
        """Seek to a specific position without destroying the window."""
        self.current_position = position_seconds
        # Set time position via IPC
        self._mpv_send(["set_property", "time-pos", float(position_seconds)])
        print(f"HD video seeked to {position_seconds:.2f}s")
    
    def pause(self, current_position=None):
        """Pause the video without destroying the window."""
        if current_position is not None:
            self.paused_at_position = current_position
        else:
            self.paused_at_position = self.current_position
        # Move to precise position and pause
        self._mpv_send(["set_property", "time-pos", float(self.paused_at_position)])
        self._mpv_send(["set_property", "pause", True])
        self.is_paused = True
        print(f"HD video paused at {self.paused_at_position:.2f}s")
    
    def resume(self):
        """Resume playback from paused position without losing the window."""
        if not self.is_paused:
            return
        self._mpv_send(["set_property", "pause", False])
        self.is_paused = False
        print(f"HD video resumed from {self.paused_at_position:.2f}s")
        
    def stop(self):
        """Stop the video player and clean up IPC."""
        if self._ipc_sock:
            try:
                # Politely ask mpv to quit
                self._mpv_send(["quit"])
            except Exception:
                pass
            try:
                self._ipc_sock.close()
            except Exception:
                pass
            self._ipc_sock = None
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
        if self.ipc_path and os.path.exists(self.ipc_path):
            try:
                os.unlink(self.ipc_path)
            except Exception:
                pass
        
    def is_running(self):
        """Check if the player is still running."""
        if self.process:
            return self.process.poll() is None
        return False


def external_video_player_worker(video_path, command_queue=None):
    """
    Worker process that uses mpv for HD video with a persistent window.
    
    Args:
        video_path: Path to the HD video file
        command_queue: Optional queue for receiving seek and pause commands
    """
    print(f"Starting HD player worker (mpv)...")
    
    player_instance = ExternalVideoPlayer(video_path)
    
    if not player_instance.start():
        print("Failed to start HD player")
        return
    
    try:
        # Monitor the player and handle commands
        running = True
        while running:
            # Check for commands if queue provided
            if command_queue:
                try:
                    if not command_queue.empty():
                        cmd = command_queue.get_nowait()
                        if cmd is None:  # Shutdown signal
                            break
                        elif cmd.get('type') == 'seek':
                            # Seek to the specified time
                            seek_time = cmd.get('time', 0)
                            player_instance.seek(seek_time)
                        elif cmd.get('type') == 'pause':
                            paused = cmd.get('paused', True)
                            current_pos = cmd.get('current_position', 0)
                            if paused:
                                player_instance.pause(current_pos)
                            else:
                                player_instance.resume()
                except Exception as e:
                    print(f"HD video command error: {e}")
            
            # Only check if running when not paused
            if not player_instance.is_paused:
                if not player_instance.is_running():
                    break
            
            time.sleep(0.1)
    finally:
        player_instance.stop()
        print("HD player worker stopped")


class VideoController:
    """
    A class to handle real-time video control with an interactive UI.
    Provides sliders for adjusting playback range and shows current position.
    """

    def __init__(self, video_path, video_stream):
        self.window_name = "Video Controls"
        cv2.namedWindow(self.window_name)

        # Store video properties
        self.video_path = video_path
        self.video_stream = video_stream
        self.fps = video_stream.fps
        self.duration = video_stream.total_frames / self.fps

        # Default range (full video)
        self.start_time = 0
        self.end_time = self.duration
        self.current_time = 0

        # Create trackbars
        cv2.createTrackbar("Start", self.window_name, 0, 1000, self._on_start_change)
        cv2.createTrackbar("End", self.window_name, 1000, 1000, self._on_end_change)

        # Create a frame for drawing
        self.height = 150
        self.width = 800
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Preview frames
        self.preview_frames = []
        self.preview_times = []
        self._load_preview_frames()

        # Create a reset button
        self.button_width = 100
        self.button_height = 30
        self.button_x = self.width - self.button_width - 10
        self.button_y = 10
        self.button_pressed = False

        # Create pause/play button
        self.play_button_width = 100
        self.play_button_height = 30
        self.play_button_x = self.button_x - self.play_button_width - 10
        self.play_button_y = 10
        self.play_button_pressed = False
        self.is_paused = False

        # Flags
        self.seeking = False
        self.reset_requested = False  # Flag for reset button
        self.last_update_time = time.time()
        self.total_duration = self.duration
        if hasattr(self.video_stream, "is_playing_sync"):
            # Add the sync sequence duration to the total duration
            self.total_duration += BLACK_SCREEN_DURATION + SYNC_SEQUENCE_DURATION

    def _load_preview_frames(self):
        """Load several preview frames from the video for the timeline."""
        try:
            # Open video to get a few representative frames
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Warning: Cannot load preview frames from {self.video_path}")
                return

            # Get small frames at intervals
            preview_count = 10
            preview_scale = 0.15

            for i in range(preview_count):
                frame_idx = int(i * self.video_stream.total_frames / preview_count)
                timestamp = frame_idx / self.fps
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Resize for preview
                    h, w = frame.shape[:2]
                    preview_width = int(w * preview_scale)
                    preview_height = int(h * preview_scale)
                    preview = cv2.resize(frame, (preview_width, preview_height))
                    self.preview_frames.append(preview)
                    self.preview_times.append(timestamp)

            cap.release()
        except Exception as e:
            print(f"Error loading preview frames: {e}")

    def _on_start_change(self, value):
        """Handle start slider movement by user."""
        # Convert normalized slider value (0-1000) to seconds
        new_start = (value / 1000.0) * self.duration

        # Ensure start < end
        if new_start >= self.end_time:
            normalized_end = int((self.end_time / self.duration) * 1000)
            cv2.setTrackbarPos("Start", self.window_name, normalized_end - 1)
            return

        # Set new start time
        if abs(new_start - self.start_time) > 0.5:  # Only seek if change is significant
            self.start_time = new_start
            self.seeking = True
            print(f"New range: {self.start_time:.1f}s - {self.end_time:.1f}s")

    def _on_end_change(self, value):
        """Handle end slider movement by user."""
        # Convert normalized slider value (0-1000) to seconds
        new_end = (value / 1000.0) * self.duration

        # Ensure end > start
        if new_end <= self.start_time:
            normalized_start = int((self.start_time / self.duration) * 1000)
            cv2.setTrackbarPos("End", self.window_name, normalized_start + 1)
            return

        # Set new end time
        if abs(new_end - self.end_time) > 0.5:  # Only update if change is significant
            self.end_time = new_end
            print(f"New range: {self.start_time:.1f}s - {self.end_time:.1f}s")

    def update(self, current_time):
        """Update UI with current playback position and handle seeking."""
        self.current_time = current_time

        # Check if we need to loop back to start
        if current_time >= self.end_time:
            self.video_stream.seek(self.start_time)
            return True  # Indicate we're looping

        # Handle seeking if range changed
        now = time.time()
        if self.seeking and (now - self.last_update_time > 0.5):  # Limit seek frequency
            self.video_stream.seek(max(0, self.start_time))
            self.seeking = False
            self.last_update_time = now
            return True  # Indicate we're seeking

        # Draw visualization frame
        self._draw_controls()

        # Check for mouse clicks
        self._handle_mouse_events()

        cv2.imshow(self.window_name, self.frame)

        return False  # No action needed

    def _handle_mouse_events(self):
        """Handle mouse events for button interaction."""

        # Get mouse position if window is in focus
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if click is within reset button
                if (
                    self.button_x <= x <= self.button_x + self.button_width
                    and self.button_y <= y <= self.button_y + self.button_height
                ):
                    self.button_pressed = True
                    # Reset the video to the very beginning (including sync sequence)
                    self.reset_requested = True  # Set flag to be handled in main loop
                    print("Reset to beginning of video requested")
                
                # Check if click is within pause/play button
                elif (
                    self.play_button_x <= x <= self.play_button_x + self.play_button_width
                    and self.play_button_y <= y <= self.play_button_y + self.play_button_height
                ):
                    self.play_button_pressed = True
                    self.is_paused = not self.is_paused
                    if self.is_paused:
                        print("Video paused")
                    else:
                        print("Video playing")

        # Set mouse callback
        cv2.setMouseCallback(self.window_name, mouse_callback)

        # Reset button states
        self.button_pressed = False
        self.play_button_pressed = False

    def _draw_controls(self):
        """Draw the control visualization with current position and range."""
        # Clear frame
        self.frame.fill(0)

        # Find nearest preview frame to current time
        nearest_idx = 0
        min_diff = float("inf")
        for i, t in enumerate(self.preview_times):
            diff = abs(t - self.current_time)
            if diff < min_diff:
                min_diff = diff
                nearest_idx = i

        # Draw preview frame if available
        if self.preview_frames and nearest_idx < len(self.preview_frames):
            preview = self.preview_frames[nearest_idx]
            h, w = preview.shape[:2]
            x_offset = (self.width - w) // 2
            y_offset = 10

            # Make sure the preview fits within the frame
            if (
                y_offset + h <= self.frame.shape[0]
                and x_offset + w <= self.frame.shape[1]
            ):
                # Only draw if it fits within frame boundaries
                self.frame[y_offset : y_offset + h, x_offset : x_offset + w] = preview
            else:
                # Resize preview to fit if needed
                max_height = self.frame.shape[0] - y_offset
                if max_height > 0:
                    scale = min(max_height / h, 1.0)
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                    if new_h > 0 and new_w > 0:
                        resized_preview = cv2.resize(preview, (new_w, new_h))
                        # Recalculate offset for centered display
                        x_offset = (self.width - new_w) // 2
                        # Make sure it fits within bounds
                        if (
                            y_offset + new_h <= self.frame.shape[0]
                            and x_offset + new_w <= self.frame.shape[1]
                        ):
                            self.frame[
                                y_offset : y_offset + new_h, x_offset : x_offset + new_w
                            ] = resized_preview

        # Draw timeline
        timeline_y = self.height - 50
        timeline_height = 20
        timeline_width = self.width - 100
        start_x = 50

        # Background track
        cv2.rectangle(
            self.frame,
            (start_x, timeline_y),
            (start_x + timeline_width, timeline_y + timeline_height),
            (60, 60, 60),
            -1,
        )

        # Calculate positions
        start_pos = start_x + int((self.start_time / self.duration) * timeline_width)
        end_pos = start_x + int((self.end_time / self.duration) * timeline_width)
        current_pos = start_x + int(
            (self.current_time / self.duration) * timeline_width
        )

        # Selected range
        cv2.rectangle(
            self.frame,
            (start_pos, timeline_y),
            (end_pos, timeline_y + timeline_height),
            (0, 180, 0),
            -1,
        )

        # Current position marker
        cv2.rectangle(
            self.frame,
            (current_pos - 2, timeline_y - 5),
            (current_pos + 2, timeline_y + timeline_height + 5),
            (0, 255, 255),
            -1,
        )

        # Format times as MM:SS
        def format_time(seconds):
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"

        # Draw time labels with MM:SS format
        cv2.putText(
            self.frame,
            format_time(self.start_time),
            (start_pos - 20, timeline_y + timeline_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.putText(
            self.frame,
            format_time(self.end_time),
            (end_pos - 20, timeline_y + timeline_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.putText(
            self.frame,
            f"Current: {format_time(self.current_time)}",
            (self.width // 2 - 50, timeline_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            1,
        )

        # Draw duration
        duration_secs = self.end_time - self.start_time
        cv2.putText(
            self.frame,
            f"Duration: {format_time(duration_secs)}",
            (self.width - 200, timeline_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        # Draw PAUSED indicator if paused
        if self.is_paused:
            # Draw a semi-transparent overlay
            paused_text = "PAUSED"
            text_size = cv2.getTextSize(paused_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
            text_x = (self.width - text_size[0]) // 2
            text_y = self.height - 80
            
            # Draw background rectangle
            padding = 10
            cv2.rectangle(
                self.frame,
                (text_x - padding, text_y - text_size[1] - padding),
                (text_x + text_size[0] + padding, text_y + padding),
                (0, 0, 0),
                -1,
            )
            
            # Draw text (bold effect with thickness 4)
            cv2.putText(
                self.frame,
                paused_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),  # Yellow
                4,  # Thickness for bold effect
                cv2.LINE_AA,
            )

        # Draw pause/play button
        play_button_color = (0, 120, 255) if self.play_button_pressed else (0, 70, 150)
        cv2.rectangle(
            self.frame,
            (self.play_button_x, self.play_button_y),
            (self.play_button_x + self.play_button_width, self.play_button_y + self.play_button_height),
            play_button_color,
            -1,
        )

        # Add pause/play button text
        play_text = "PLAY" if self.is_paused else "PAUSE"
        play_text_offset = 20 if self.is_paused else 15
        cv2.putText(
            self.frame,
            play_text,
            (self.play_button_x + play_text_offset, self.play_button_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Draw reset button
        button_color = (0, 120, 255) if self.button_pressed else (0, 70, 150)
        cv2.rectangle(
            self.frame,
            (self.button_x, self.button_y),
            (self.button_x + self.button_width, self.button_y + self.button_height),
            button_color,
            -1,
        )

        # Add button text
        cv2.putText(
            self.frame,
            "RESET",
            (self.button_x + 20, self.button_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    def get_range(self):
        """Get current start and end times."""
        return self.start_time, self.end_time

    def is_video_paused(self):
        """Check if video is currently paused."""
        return self.is_paused
    
    def check_reset_requested(self):
        """Check if reset was requested and clear the flag."""
        if self.reset_requested:
            self.reset_requested = False
            return True
        return False


class VideoStream:
    """
    A class to handle video streaming in a separate process.
    Provides a queue-based interface for frames.
    """

    def __init__(self, video_path, start_time=0, end_time=None, buffer_size=30):
        self.video_path = video_path
        self.start_time = start_time
        self.end_time = end_time
        self.buffer_size = buffer_size

        # Create queues
        self.frame_queue = multiprocessing.Queue(maxsize=buffer_size)
        self.command_queue = multiprocessing.Queue()

        # Get video properties before starting worker
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Start the worker process
        self.process = multiprocessing.Process(
            target=self._stream_worker,
            args=(
                self.video_path,
                self.start_time,
                self.end_time,
                self.frame_queue,
                self.command_queue,
                self.fps,  # Pass fps to worker
            ),
        )
        self.process.daemon = True
        self.process.start()

    @staticmethod
    def _stream_worker(
        video_path, start_time, end_time, frame_queue, command_queue, fps
    ):
        """Worker process that reads frames and pushes them to the queue."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        # Set initial position
        if start_time > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        # Enforce strict frame timing
        frame_duration = 1.0 / fps if fps > 0 else 0.033
        next_frame_time = time.time() + frame_duration

        running = True
        paused = False
        last_frame = None
        last_metadata = None
        
        while running:
            # Check for commands
            try:
                if not command_queue.empty():
                    cmd = command_queue.get_nowait()
                    if cmd["type"] == "seek":
                        cap.set(cv2.CAP_PROP_POS_MSEC, cmd["time"] * 1000)
                        # Reset timing after seek
                        next_frame_time = time.time() + frame_duration
                        last_frame = None
                        last_metadata = None
                    elif cmd["type"] == "stop":
                        running = False
                    elif cmd["type"] == "pause":
                        paused = cmd.get("paused", True)
                        if not paused:
                            # Reset timing when unpausing
                            next_frame_time = time.time() + frame_duration
            except Exception:
                pass

            # If paused, keep sending the last frame
            if paused:
                if last_frame is not None and last_metadata is not None:
                    try:
                        if frame_queue.full():
                            try:
                                frame_queue.get_nowait()
                            except Exception:
                                pass
                        frame_queue.put((last_frame.copy(), last_metadata.copy()), block=False)
                    except Exception:
                        pass
                time.sleep(0.1)
                continue

            # Wait until it's time for the next frame
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Read frame
            ret, frame = cap.read()

            if not ret:
                # End of video, loop back to start_time
                cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
                next_frame_time = time.time() + frame_duration
                continue

            # Check if we've reached end_time
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if end_time is not None and current_time >= end_time:
                # Loop back to start_time
                cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
                next_frame_time = time.time() + frame_duration
                continue

            # Store frame for pause functionality
            last_frame = frame
            last_metadata = {
                "timestamp": current_time,
                "frame_number": cap.get(cv2.CAP_PROP_POS_FRAMES),
            }

            # Try to add frame to queue, skip if full
            try:
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()  # Remove oldest frame
                    except Exception:
                        pass
                frame_queue.put((frame, last_metadata.copy()), block=False)
            except Exception:
                pass

            # Set time for next frame
            next_frame_time += frame_duration

        cap.release()

    def read(self, timeout=1.0):
        """Read the next frame from the stream."""
        try:
            frame_data = self.frame_queue.get(timeout=timeout)
            return True, frame_data[0], frame_data[1]
        except Exception:
            return False, None, None

    def seek(self, time_seconds):
        """Seek to a specific time in the video."""
        self.command_queue.put({"type": "seek", "time": time_seconds})

        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                break

    def pause(self, paused=True):
        """Pause or unpause the video stream."""
        self.command_queue.put({"type": "pause", "paused": paused})

    def stop(self):
        """Stop the streaming process."""
        self.command_queue.put({"type": "stop"})
        self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.terminate()


def generate_sync_sequence(
    fps,
    width,
    height,
    duration=SYNC_SEQUENCE_DURATION,
    black_screen_duration=BLACK_SCREEN_DURATION,
    post_convergence_pause=POST_CONVERGENCE_PAUSE,
):
    """
    Generate a synchronization sequence consisting of:
    1. A black screen for the specified duration
    2. White circles moving in different patterns against a high-tech background with axes
    3. Circles converging to the center
    4. A pause with circles at the center before transitioning to main video

    Features:
    - Initial black screen for audio/video preparation
    - Transparent white grid background with fine scientific grid
    - Gradual appearance of circles with sinusoidal brightness variation
    - Converging animation at the end
    - Pause after convergence with circles at the center
    - No text overlay for clean visual appearance

    Parameters:
    - fps: frames per second
    - width: frame width
    - height: frame height
    - duration: duration of the sync pattern in seconds (default: 30 seconds)
    - black_screen_duration: duration of initial black screen in seconds (default: 5 seconds)
    - post_convergence_pause: duration of pause after convergence in seconds (default: 5 seconds)

    Returns:
    - List of numpy array frames with metadata for NUM_SHAPES control
    """
    print(
        f"Generating sync sequence: {black_screen_duration}s black + {duration}s pattern + {post_convergence_pause}s pause..."
    )

    # Calculate total number of frames
    black_frame_count = int(fps * black_screen_duration)
    pattern_frame_count = int(fps * duration)
    pause_frame_count = int(fps * post_convergence_pause)
    total_frame_count = black_frame_count + pattern_frame_count + pause_frame_count
    frames = []

    # 1. Generate black screen frames
    for i in range(black_frame_count):
        # Create a black canvas
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # No active circles during black screen
        frame_metadata = {"num_shapes": 0}
        frames.append((frame, frame_metadata))

        # Progress indicator
        if i % int(black_frame_count / 5) == 0 and i > 0:
            percent = int(100 * i / black_frame_count)
            print(f"Generating black screen: {percent}% complete")

    print("Black screen generation complete.")

    # 2. Generate pattern frames

    # Circle properties
    circle_radius = min(width, height) // 20
    base_circle_color = (255, 255, 255)  # Base white color that will be modulated

    # Direction axes for each circle - REDUCED TO 3 CIRCLES
    # 1: North-South
    # 2: East-West
    # 3: 45-degree diagonal axis
    directions = [
        (0, 1),  # North-South
        (1, 0),  # East-West
        (1, 1),  # 45 degrees (diagonal)
    ]

    # Calculate center of the frame
    center_x = width // 2
    center_y = height // 2

    # Maximum amplitude of the movement (how far from center)
    amplitude = min(width, height) // 4

    # Frame for each circle to appear
    circle_appear_frame = [
        0,  # First circle appears immediately
        int(pattern_frame_count * 0.33),  # Second circle appears at 33% of sequence
        int(pattern_frame_count * 0.66),  # Third circle appears at 66% of sequence
    ]

    # Frame when convergence animation begins (5 seconds before end of pattern)
    convergence_start_frame = pattern_frame_count - int(fps * 5)

    # Create pattern frames
    for frame_idx in range(pattern_frame_count):
        # Create a black canvas with high-tech background
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Calculate time progress (0.0 to 1.0)
        time_progress = frame_idx / pattern_frame_count

        # Draw the high-tech background with axes - using transparent white
        # Main axes (horizontal and vertical)
        cv2.line(frame, (0, center_y), (width, center_y), (30, 30, 30), 1)
        cv2.line(frame, (center_x, 0), (center_x, height), (30, 30, 30), 1)

        # Secondary horizontal lines - more fine-grained
        for i in range(1, 21):  # Increased from 11 to 21 lines
            y_offset = height * i // 20
            cv2.line(frame, (0, y_offset), (width, y_offset), (15, 15, 15), 1)

        # Secondary vertical lines - more fine-grained
        for i in range(1, 21):  # Increased from 11 to 21 lines
            x_offset = width * i // 20
            cv2.line(frame, (x_offset, 0), (x_offset, height), (15, 15, 15), 1)

        # Grid for high-tech background - more fine-grained scientific
        grid_spacing = min(width, height) // 80  # Reduced from 40 to 80 for finer grid
        for i in range(0, width, grid_spacing):
            for j in range(0, height, grid_spacing):
                # Use smaller dots for the grid
                cv2.circle(frame, (i, j), 1, (10, 10, 10), 1)

        # Draw 45-degree diagonal lines
        cv2.line(frame, (0, 0), (width, height), (20, 20, 20), 1)
        cv2.line(frame, (width, 0), (0, height), (20, 20, 20), 1)

        # Calculate how many circles should be visible at this frame
        active_circles = sum(
            1 for i in range(len(directions)) if frame_idx >= circle_appear_frame[i]
        )
        num_shapes_value = (
            active_circles  # Update NUM_SHAPES value based on visible circles
        )

        # Track if we're in the convergence phase
        in_convergence = frame_idx >= convergence_start_frame
        convergence_progress = 0
        if in_convergence:
            # Calculate convergence progress from 0 to 1
            convergence_progress = (frame_idx - convergence_start_frame) / (
                pattern_frame_count - convergence_start_frame
            )

        # Draw rotating circles with sinusoidal brightness
        for i, direction in enumerate(directions):
            if frame_idx >= circle_appear_frame[i]:
                # Calculate how long this circle has been visible (for fade-in effect)
                frames_visible = frame_idx - circle_appear_frame[i]
                fade_in_duration = int(fps * 1.5)  # 1.5 seconds fade-in
                fade_in_factor = min(1.0, frames_visible / fade_in_duration)

                # Calculate phase shift based on circle index and frame
                phase_shift = 2 * math.pi * i / len(directions)

                # Calculate position using sine wave, but converge to center if in convergence phase
                t = (
                    2 * math.pi * frame_idx / (fps * 5)
                )  # 5-second period for oscillation

                # Generate a sinusoidal brightness oscillation (from 0.6 to 1.0)
                brightness_oscillation = 0.6 + 0.4 * (
                    0.5 + 0.5 * math.sin(t * 0.5 + phase_shift)
                )

                if in_convergence:
                    # Gradually reduce amplitude to make circles converge to center
                    convergence_factor = 1.0 - convergence_progress
                    curr_amplitude = amplitude * convergence_factor
                else:
                    curr_amplitude = amplitude

                dx = curr_amplitude * direction[0] * math.sin(t + phase_shift)
                dy = curr_amplitude * direction[1] * math.sin(t + phase_shift)

                # Calculate circle position
                x = int(center_x + dx)
                y = int(center_y + dy)

                # Apply brightness oscillation and fade-in to the circle color
                # Map from light gray (153) to white (255)
                min_brightness = 153  # Light gray
                brightness_range = 255 - min_brightness

                # Calculate brightness oscillation - make sure we use cosine for smoother variation
                # and ensure a larger amplitude of variation
                brightness_oscillation = 0.5 + 0.5 * math.cos(t * 0.7 + phase_shift * 2)

                # Combine fade-in with brightness oscillation
                current_brightness = (
                    min_brightness
                    + brightness_range * brightness_oscillation * fade_in_factor
                )

                # Ensure brightness is an integer within the valid range
                current_brightness = int(
                    max(min_brightness, min(255, current_brightness))
                )

                # Create the color tuple with the calculated brightness
                circle_color_modulated = (
                    current_brightness,
                    current_brightness,
                    current_brightness,
                )

                # Draw circle with modulated brightness
                cv2.circle(frame, (x, y), circle_radius, circle_color_modulated, -1)

                # Show movement trace (trail), but not during convergence
                if frame_idx > 5 and not in_convergence:
                    # Calculate 5 previous positions
                    for back in range(1, 6):
                        if frame_idx - back >= circle_appear_frame[i]:
                            prev_t = 2 * math.pi * (frame_idx - back) / (fps * 5)
                            prev_dx = (
                                amplitude
                                * direction[0]
                                * math.sin(prev_t + phase_shift)
                            )
                            prev_dy = (
                                amplitude
                                * direction[1]
                                * math.sin(prev_t + phase_shift)
                            )
                            prev_x = int(center_x + prev_dx)
                            prev_y = int(center_y + prev_dy)

                            # Draw fading trail points with reduced brightness from the main circle
                            fade_factor = (0.7 - (back * 0.1)) * fade_in_factor
                            trail_brightness = int(
                                max(
                                    min_brightness,
                                    min(255, current_brightness * fade_factor),
                                )
                            )
                            trail_color = (
                                trail_brightness,
                                trail_brightness,
                                trail_brightness,
                            )
                            cv2.circle(frame, (prev_x, prev_y), 2, trail_color, -1)

        # Store frame with metadata for NUM_SHAPES control
        frame_metadata = {"num_shapes": num_shapes_value}
        frames.append((frame, frame_metadata))

        # Progress indicator
        if frame_idx % int(pattern_frame_count / 10) == 0:
            percent = int(100 * frame_idx / pattern_frame_count)
            print(f"Generating pattern sequence: {percent}% complete")

    # Create post-convergence pause frames (circles stay at center)
    print("Generating post-convergence pause frames...")
    for i in range(pause_frame_count):
        # Create a black canvas with high-tech background
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw the same grid background as in the pattern frames
        # Main axes
        cv2.line(frame, (0, center_y), (width, center_y), (30, 30, 30), 1)
        cv2.line(frame, (center_x, 0), (center_x, height), (30, 30, 30), 1)

        # Secondary horizontal lines
        for j in range(1, 21):
            y_offset = height * j // 20
            cv2.line(frame, (0, y_offset), (width, y_offset), (15, 15, 15), 1)

        # Secondary vertical lines
        for j in range(1, 21):
            x_offset = width * j // 20
            cv2.line(frame, (x_offset, 0), (x_offset, height), (15, 15, 15), 1)

        # Grid
        grid_spacing = min(width, height) // 80
        for j in range(0, width, grid_spacing):
            for k in range(0, height, grid_spacing):
                cv2.circle(frame, (j, k), 1, (10, 10, 10), 1)

        # Diagonal lines
        cv2.line(frame, (0, 0), (width, height), (20, 20, 20), 1)
        cv2.line(frame, (width, 0), (0, height), (20, 20, 20), 1)

        # All circles are at the center, fully visible
        for idx, _ in enumerate(directions):
            # Calculate brightness - continue the oscillation from where pattern frames left off
            t = 2 * math.pi * (pattern_frame_count + i) / (fps * 5)
            brightness_oscillation = 0.5 + 0.5 * math.cos(
                t * 0.7 + 2 * math.pi * idx / len(directions)
            )

            # Calculate brightness value
            min_brightness = 153
            brightness_range = 255 - min_brightness
            current_brightness = int(
                min_brightness + brightness_range * brightness_oscillation
            )

            # Draw the circle at the center with the calculated brightness
            circle_color = (current_brightness, current_brightness, current_brightness)
            cv2.circle(frame, (center_x, center_y), circle_radius, circle_color, -1)

        # All circles are visible
        frame_metadata = {"num_shapes": len(directions)}
        frames.append((frame, frame_metadata))

        # Progress indicator
        if i % int(pause_frame_count / 5) == 0 and i > 0:
            percent = int(100 * i / pause_frame_count)
            print(f"Generating pause frames: {percent}% complete")

    print(
        f"Sync sequence generation complete. Total frames: {len(frames)} ({len(frames)/fps:.1f}s)"
    )
    return frames


def export_calibration_video(
    width=3840, height=2160, fps=60, output_path="calibration_4k.mp4"
):
    """
    Export the calibration sequence as a standalone 4K 60fps video file.

    Parameters:
    - width: Width of the output video (default: 3840 for 4K)
    - height: Height of the output video (default: 2160 for 4K)
    - fps: Frames per second (default: 60)
    - output_path: Path to save the video file

    Returns:
    - Path to the saved video file
    """
    print(
        f"Exporting calibration sequence as {width}x{height}@{fps}fps to {output_path}..."
    )

    # Generate the calibration sequence frames
    frames = generate_sync_sequence(fps, width, height)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'avc1' for H.264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create video writer for {output_path}")
        return None

    # Write frames to the video file
    frame_count = len(frames)
    for i, (frame, _) in enumerate(frames):
        out.write(frame)

        # Progress indicator
        if i % (frame_count // 10) == 0:
            print(f"Writing video: {i}/{frame_count} frames ({i/frame_count*100:.1f}%)")

    # Release resources
    out.release()
    print(f"Calibration video exported to {output_path}")

    return output_path


class SyncVideoStream(VideoStream):
    """
    Extension of VideoStream that prepends a sync sequence before the actual video content.
    Provides max_shapes value in metadata based on visible circles in the sync sequence.
    """

    def __init__(self, video_path, start_time=0, end_time=None, buffer_size=30):
        super().__init__(video_path, start_time, end_time, buffer_size)

        # Store original NUM_SHAPES value for reference
        self.original_num_shapes = NUM_SHAPES

        # Generate the sync sequence frames
        self.sync_frames = generate_sync_sequence(self.fps, self.width, self.height)
        self.total_sync_frames = len(self.sync_frames)
        self.sync_frame_index = 0
        self.is_playing_sync = True
        self.sync_start_time = time.time()
        self.sync_frame_duration = 1.0 / self.fps
        
        # Pause support for sync sequence
        self.sync_paused = False
        self.sync_paused_elapsed = 0  # Time elapsed when paused

        # Update total_frames to include sync sequence
        self.original_total_frames = self.total_frames
        self.total_frames += self.total_sync_frames

        # Calculate total sync duration
        self.total_sync_duration = (
            BLACK_SCREEN_DURATION + SYNC_SEQUENCE_DURATION + POST_CONVERGENCE_PAUSE
        )

    def read(self, timeout=1.0):
        """Read the next frame from sync sequence or stream. Returns max_shapes in metadata."""
        if self.is_playing_sync:
            # Calculate what sync frame index we should be on based on elapsed time
            current_time = time.time()
            if self.sync_paused:
                # Use the paused elapsed time
                elapsed = self.sync_paused_elapsed
            else:
                elapsed = current_time - self.sync_start_time
            target_frame = int(elapsed * self.fps)

            # If we've reached the end of sync sequence
            if target_frame >= self.total_sync_frames:
                self.is_playing_sync = False
                # Explicitly seek to the beginning of the actual video
                super().seek(0)
                # Small delay to allow the stream worker to process the seek
                time.sleep(0.05)
                return super().read(timeout)

            # Return the appropriate sync frame
            frame, metadata = self.sync_frames[target_frame]

            timestamp = elapsed
            frame_metadata = {
                "timestamp": timestamp,
                "frame_number": target_frame,
                "is_sync": True,
                "max_shapes": metadata["num_shapes"],  # Pass max_shapes in metadata instead of modifying global
            }
            return True, frame, frame_metadata
        else:
            # Use parent class to read from the actual video
            ret, frame, metadata = super().read(timeout)
            if ret and metadata:
                # Add max_shapes to metadata for consistency (use default NUM_SHAPES for regular video)
                metadata["max_shapes"] = self.original_num_shapes
            return ret, frame, metadata

    def pause(self, paused=True):
        """Override pause to handle sync sequence."""
        if self.is_playing_sync:
            if paused:
                # Store elapsed time when pausing during sync
                self.sync_paused = True
                self.sync_paused_elapsed = time.time() - self.sync_start_time
            else:
                # Resume from paused state
                self.sync_paused = False
                # Adjust sync_start_time to account for pause duration
                self.sync_start_time = time.time() - self.sync_paused_elapsed
        # Also call parent pause for the actual video stream
        super().pause(paused)

    def seek(self, time_seconds):
        """
        Seek to a specific time in the video or sync sequence.
        If time is less than sync sequence duration, seek within sync.
        Otherwise, seek to (time - sync_duration) in the actual video.
        """
        if time_seconds < self.total_sync_duration:
            # Seek within sync sequence
            self.is_playing_sync = True
            self.sync_start_time = time.time() - time_seconds
            self.sync_paused = False  # Clear pause state on seek
            return True
        else:
            # Seeking to the main video
            self.is_playing_sync = False
            self.sync_paused = False  # Clear pause state
            # Seek in the actual video (accounting for full sync duration)
            actual_video_time = time_seconds - self.total_sync_duration
            return super().seek(actual_video_time)


def main(
    video_path=None,
    video_hd_path=None,
    midi_mapping_path="midi_mapping_analog_four_mk2.csv",
    debug=False,
    export_calibration=False,
    export_resolution=(3840, 2160),
    export_fps=60,
    skip_calibration=False,  # Add parameter to make calibration optional
):
    global DEBUG
    DEBUG = debug

    print("Starting program...")
    print(f"Video path: {video_path}")
    print(f"HD Video path: {video_hd_path}")
    print(f"MIDI mapping: {midi_mapping_path}")
    print(f"Skip calibration: {skip_calibration}")
    if video_hd_path:
        print(f"HD video mode: mpv")

    # Check if export calibration was requested
    if export_calibration:
        height, width = export_resolution
        output_path = "calibration_4k.mp4"
        print(f"Exporting calibration sequence at {width}x{height}@{export_fps}fps...")
        export_path = export_calibration_video(width, height, export_fps, output_path)
        if export_path:
            print(f"Calibration sequence exported to {export_path}")
        else:
            print("Failed to export calibration sequence")
        return

    # Check if video exists
    if video_path and not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Check if HD video exists
    if video_hd_path and not os.path.exists(video_hd_path):
        print(f"Error: HD video file not found at {video_hd_path}")
        return
    
    # Check if mpv is available when using HD video
    if video_hd_path:
        has_mpv = shutil.which('mpv') is not None
        if not has_mpv:
            print(f"\nError: mpv not found!")
            print(f"mpv is required for HD video playback.")
            print(f"Please install mpv:")
            print(f"  macOS: brew install mpv")
            print(f"  Linux: sudo apt-get install mpv\n")
            return
    
    # Compare video lengths if both videos are provided
    if video_path and video_hd_path:
        print("Comparing video lengths...")
        cap1 = cv2.VideoCapture(video_path)
        cap2 = cv2.VideoCapture(video_hd_path)
        
        if cap1.isOpened() and cap2.isOpened():
            fps1 = cap1.get(cv2.CAP_PROP_FPS)
            fps2 = cap2.get(cv2.CAP_PROP_FPS)
            frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            
            duration1 = frame_count1 / fps1 if fps1 > 0 else 0
            duration2 = frame_count2 / fps2 if fps2 > 0 else 0
            
            print(f"Main video: {duration1:.2f}s ({frame_count1} frames @ {fps1:.2f} fps)")
            print(f"HD video: {duration2:.2f}s ({frame_count2} frames @ {fps2:.2f} fps)")
            
            if abs(duration1 - duration2) > 1.0:  # Allow 1 second difference
                print(f"WARNING: Video lengths differ by {abs(duration1 - duration2):.2f} seconds!")
                print("Videos may not stay in sync during playback.")
            else:
                print("Video lengths are compatible.")
        
        cap1.release()
        cap2.release()

    try:
        midi_out = init_midi_output()

        # Load MIDI mapping
        try:
            midi_mapping = pd.read_csv(midi_mapping_path)
            if DEBUG:
                debug_print(midi_mapping)
        except Exception as e:
            print(f"Error loading MIDI mapping: {e}")
            return

        # Video file is required
        if not video_path:
            print("Error: Video file is required. Use --video to specify a video file.")
            return
        
        try:
            print("Opening video file...")
            # Verify the video can be opened before proceeding
            cap_test = cv2.VideoCapture(video_path)
            if not cap_test.isOpened():
                print(f"Error: Cannot open video file at {video_path}")
                return
            fps = cap_test.get(cv2.CAP_PROP_FPS)
            print(f"Video FPS: {fps}")
            cap_test.release()

            # Initialize video stream with full range initially and sync sequence
            if skip_calibration:
                print("Skipping calibration sequence...")
                video_stream = VideoStream(video_path, 0, None)
            else:
                print("Initializing video stream with sync sequence...")
                video_stream = SyncVideoStream(video_path, 0, None)
            frame_time = 1.0 / fps if fps > 0 else 0.033
            print(f"Target frame time: {frame_time:.4f} seconds")
            
            # Immediately pause the video stream to prevent it from starting before mpv is ready
            print("Pausing video stream until both players are ready...")
            video_stream.pause(paused=True)

            # Create persistent controller window
            controller = VideoController(video_path, video_stream)

        except Exception as e:
            import traceback

            print(f"Error setting up video: {e}")
            print(traceback.format_exc())
            return

        def signal_handler(sig, frame):
            print("Ctrl+C pressed, exiting...")
            send_stop_message(midi_out)
            if "video_stream" in locals():
                video_stream.stop()
            cv2.destroyAllWindows()
            if midi_out:
                midi_out.close()
            exit(0)

        # Register signal handler for graceful shutdown.
        signal.signal(signal.SIGINT, signal_handler)

        # Set up the visualization process with a small maxsize Queue.
        print("Creating visualization process...")
        viz_queue = multiprocessing.Queue(maxsize=1)
        viz_process = multiprocessing.Process(
            target=visualization_worker, args=(viz_queue,)
        )
        viz_process.daemon = True
        viz_process.start()
        
        # Set up the HD visualization process with ffplay if HD video is provided
        hd_viz_process = None
        hd_player_command_queue = None
        
        # Calculate sync offset for HD video (if using sync sequence)
        sync_offset = 0
        if video_path and not skip_calibration and hasattr(video_stream, 'total_sync_duration'):
            sync_offset = video_stream.total_sync_duration
            print(f"HD video will be offset by {sync_offset:.2f}s to account for sync sequence")
        
        if video_hd_path:
            # Use mpv for HD video playback
            print(f"Creating HD visualization process with mpv...")
            hd_player_command_queue = multiprocessing.Queue()
            hd_viz_process = multiprocessing.Process(
                target=external_video_player_worker,
                args=(video_hd_path, hd_player_command_queue)
            )
            hd_viz_process.daemon = True
            hd_viz_process.start()
            time.sleep(1.5)  # Give player time to fully start and be ready
            
            # Always pause, seek to 0, and then unpause/keep paused for proper sync
            # This ensures the video is at exactly position 0 before playback
            print("Synchronizing HD video to position 0...")
            hd_player_command_queue.put({"type": "pause", "paused": True, "current_position": 0})
            time.sleep(0.2)  # Wait for pause to complete
            hd_player_command_queue.put({"type": "seek", "time": 0})
            time.sleep(0.3)  # Wait for seek to complete
            
            if sync_offset > 0:
                # Keep HD video paused during sync sequence
                print("HD video synced to position 0 and paused during sync sequence")
            else:
                # Unpause HD video to start playback (now properly synced at 0)
                hd_player_command_queue.put({"type": "pause", "paused": False})
                print("HD video synced to position 0 and ready for playback")
            
            # Now that HD video is ready, unpause the Python video stream (if using video file)
            # Both players are now synchronized and ready to start together
            if video_path:
                print("Both players ready - starting synchronized playback...")
                video_stream.pause(paused=False)
        else:
            # No HD video, so unpause the Python video stream immediately (if using video file)
            if video_path:
                print("No HD video - starting playback...")
                video_stream.pause(paused=False)

        # Initialize the recognized shapes history.
        active_shapes = []

        print("Starting main loop...")
        last_frame_time = time.time()
        start_time = time.time()  # Track overall elapsed time
        frame_count = 0  # Count frames for timing verification
        last_paused_state = False  # Track pause state changes
        last_video_timestamp = 0  # Track previous timestamp to detect seeks
        hd_video_started = (sync_offset == 0)  # Track if HD video has started playing

        while True:
            frame_count += 1

            # Get current frame from video
            ret, frame, metadata = video_stream.read()
            if ret:
                video_timestamp = metadata["timestamp"]

                # Check if reset was requested
                if controller and controller.check_reset_requested():
                    print("Processing reset request...")
                    # Reset controller's start time to 0
                    controller.start_time = 0
                    # Update the trackbar to reflect the reset
                    cv2.setTrackbarPos("Start", controller.window_name, 0)
                    
                    # Reset main video to beginning (including sync sequence if applicable)
                    if hasattr(video_stream, "is_playing_sync"):
                        video_stream.is_playing_sync = True
                        video_stream.sync_start_time = time.time()
                        video_stream.sync_paused = False
                    else:
                        video_stream.seek(0)
                    
                    # Reset HD video to position 0
                    if hd_player_command_queue:
                        hd_player_command_queue.put({"type": "seek", "time": 0})
                        # If sync sequence is active, pause HD video
                        if sync_offset > 0:
                            time.sleep(0.05)  # Small delay to allow seek to complete
                            hd_player_command_queue.put({"type": "pause", "paused": True, "current_position": 0})
                            hd_video_started = False  # Reset flag so HD video will start when sync completes
                    
                    # Reset timing
                    last_frame_time = time.time()
                    last_video_timestamp = 0
                    print("Reset complete")
                    continue

                # Check if we've passed the sync sequence and need to start HD video
                if not hd_video_started and video_timestamp >= sync_offset and hd_player_command_queue:
                    # Ensure HD video is at position 0 before starting playback
                    hd_player_command_queue.put({"type": "seek", "time": 0})
                    time.sleep(0.05)  # Small delay to allow seek to complete
                    hd_player_command_queue.put({"type": "pause", "paused": False})
                    print(f"Starting HD video playback (sync sequence complete, synced to position 0)")
                    hd_video_started = True

                # Check if pause state changed
                if controller:
                    current_paused_state = controller.is_video_paused()
                    if current_paused_state != last_paused_state:
                        # Pause state changed, update video streams
                        video_stream.pause(current_paused_state)
                        if hd_player_command_queue:
                            # Calculate HD video position (accounting for sync offset)
                            hd_video_position = max(0, video_timestamp - sync_offset)
                            hd_player_command_queue.put({
                                "type": "pause", 
                                "paused": current_paused_state,
                                "current_position": hd_video_position
                            })
                        last_paused_state = current_paused_state

                # Detect significant timestamp jumps (seeking) - more than 0.5 seconds difference
                timestamp_diff = abs(video_timestamp - last_video_timestamp)
                if timestamp_diff > 0.5 and frame_count > 2:  # Skip first few frames
                    # A seek happened, sync HD video (only if past sync sequence)
                    if hd_player_command_queue and video_timestamp >= sync_offset:
                        # Calculate HD video position (accounting for sync offset)
                        hd_video_position = video_timestamp - sync_offset
                        hd_player_command_queue.put({"type": "seek", "time": hd_video_position})
                        print(f"Syncing HD video to {hd_video_position:.2f}s (main video at {video_timestamp:.2f}s)")
                
                last_video_timestamp = video_timestamp

                # Update controller and check if it requested a seek operation
                if controller and controller.update(video_timestamp):
                    # Controller requested seek, so reset timing
                    last_frame_time = time.time()
                    continue
            else:
                print("Warning: Couldn't read frame")
                time.sleep(0.1)
                continue

            # Process the frame (even when paused, we still process to keep display updated)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Get max_shapes from metadata if available, otherwise use default
            max_shapes = metadata.get("max_shapes", NUM_SHAPES) if metadata else NUM_SHAPES
            
            # Only update MIDI when not paused
            if not (controller and controller.is_video_paused()):
                active_shapes = process_frame(
                    midi_out, gray_frame, active_shapes, midi_mapping, max_shapes=max_shapes
                )

                # Apply ambient effects based on average brightness
                send_ambient_effects(
                    midi_out, gray_frame, midi_mapping, channel=MIDI_CHANNEL
                )

            # Prepare shape data for visualization
            shapes_info = []
            for shape in active_shapes:
                shape_data = {
                    'centroid': shape.centroid,
                    'is_active': shape.is_active,
                    'channel': shape.channel,
                    'shape_counter': shape.shape_counter,
                    'num_notes': shape.num_notes,
                    'contour': shape.contour
                }
                shapes_info.append(shape_data)

            # Send the frame and shape info to the visualization process
            try:
                if viz_queue.full():
                    try:
                        viz_queue.get_nowait()
                    except Exception:
                        pass
                viz_queue.put_nowait((frame, shapes_info))
            except Exception as e:
                print(f"Visualization queue put error: {e}")

            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                print("ESC pressed, exiting...")
                break

            # Every 100 frames, check overall timing performance
            if frame_count % 100 == 0 and DEBUG:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                expected_fps = fps
                debug_print(
                    f"Overall FPS: {actual_fps:.2f} (Expected: {expected_fps:.2f})"
                )
                if abs(actual_fps - expected_fps) > 1.0:
                    debug_print("WARNING: Playback speed differs from expected!")

            # For real-time processing, we don't need to sleep here - the VideoStream
            # worker already enforces proper timing between frames

    except Exception as e:
        import traceback

        print(f"Unexpected error: {e}")
        print(traceback.format_exc())

    finally:
        print("Cleaning up resources...")
        if "video_stream" in locals():
            video_stream.stop()
        cv2.destroyAllWindows()
        if "midi_out" in locals() and midi_out:
            midi_out.close()
        if "viz_queue" in locals() and "viz_process" in locals():
            try:
                viz_queue.put(None, timeout=1)
            except Exception:
                pass
            viz_process.join(timeout=1)
        if "hd_player_command_queue" in locals() and hd_player_command_queue and "hd_viz_process" in locals() and hd_viz_process:
            try:
                hd_player_command_queue.put(None, timeout=1)
            except Exception:
                pass
            hd_viz_process.join(timeout=1)
        print("Program finished.")


def visualization_worker(frame_queue):
    """
    Worker process for visualizing detected shapes.
    Receives frames and shape info through a queue.
    """
    try:
        print("Visualization process started")
        cv2.namedWindow("Shapes Detection", cv2.WINDOW_NORMAL)

        while True:
            try:
                data = frame_queue.get(timeout=1.0)
                if data is None:
                    break

                frame, shapes_info = data

                # Draw shapes on the frame with improved visualization
                for shape_data in shapes_info:
                    centroid = shape_data['centroid']
                    is_active = shape_data['is_active']
                    channel = shape_data['channel']
                    shape_counter = shape_data['shape_counter']
                    num_notes = shape_data['num_notes']
                    contour = shape_data['contour']
                    
                    if is_active:
                        # Get color for this channel
                        color = CHANNEL_COLORS.get(channel, (255, 255, 255))  # Default to white if channel not found
                        
                        # Draw shape overlay with transparency
                        if contour is not None:
                            # Create overlay with channel color and transparency
                            overlay = frame.copy()
                            cv2.drawContours(overlay, [contour], -1, color, thickness=-1)
                            
                            # Blend overlay with original frame (30% transparency)
                            alpha = 0.3
                            frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
                        
                        # Draw circle at centroid with channel color
                        cv2.circle(
                            frame, 
                            (int(centroid[0]), int(centroid[1])), 
                            20, 
                            color, 
                            -1
                        )
                        
                        # Display shape counter in the center of the circle
                        counter_text = str(shape_counter)
                        text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        text_x = int(centroid[0] - text_size[0] // 2)
                        text_y = int(centroid[1] + text_size[1] // 2)
                        
                        # Draw counter text with black outline for better visibility
                        cv2.putText(
                            frame,
                            counter_text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),  # Black outline
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            counter_text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),  # White text
                            1,
                            cv2.LINE_AA,
                        )
                        
                        # Display num_notes below the counter
                        notes_text = f"num_notes={num_notes}"
                        notes_text_size = cv2.getTextSize(notes_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        notes_text_x = int(centroid[0] - notes_text_size[0] // 2)
                        notes_text_y = int(centroid[1] + text_size[1] + 15)  # Position below counter
                        
                        # Draw notes text with black outline for better visibility
                        cv2.putText(
                            frame,
                            notes_text,
                            (notes_text_x, notes_text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 0),  # Black outline
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            notes_text,
                            (notes_text_x, notes_text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),  # White text
                            1,
                            cv2.LINE_AA,
                        )

                cv2.imshow("Shapes Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break

            except queue.Empty:
                # No frame available, continue waiting
                continue
            except Exception as e:
                print(f"Visualization error: {e}")
                break

        cv2.destroyWindow("Shapes Detection")
        print("Visualization process ended")
    except Exception as e:
        import traceback

        print(f"Visualization process error: {e}")
        print(traceback.format_exc())


# Make sure we have all required imports at the top of the file
import os
import queue
import traceback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video or webcam feed to MIDI")
    parser.add_argument(
        "--video", type=str, help="Path to video file (leave blank for webcam)"
    )
    parser.add_argument(
        "--video-hd", "--video_hd", type=str, help="Path to high-resolution video file (optional)"
    )
    parser.add_argument(
        "--midi-mapping",
        type=str,
        default="midi_mapping_analog_four_mk2.csv",
        help="Path to MIDI mapping CSV file",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--export-calibration",
        action="store_true",
        help="Export the calibration sequence as a 4K 60fps video",
    )
    parser.add_argument(
        "--export-width",
        type=int,
        default=3840,
        help="Width of exported calibration video (default: 3840 for 4K)",
    )
    parser.add_argument(
        "--export-height",
        type=int,
        default=2160,
        help="Height of exported calibration video (default: 2160 for 4K)",
    )
    parser.add_argument(
        "--export-fps",
        type=int,
        default=60,
        help="Frame rate of exported calibration video (default: 60fps)",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip the calibration sequence when processing video",
    )
    args = parser.parse_args()

    # Print command line arguments
    print("Command line arguments:")
    print(f"  video: {args.video}")
    print(f"  video-hd: {args.video_hd}")
    print(f"  midi-mapping: {args.midi_mapping}")
    print(f"  debug: {args.debug}")
    print(f"  export-calibration: {args.export_calibration}")
    print(f"  skip-calibration: {args.skip_calibration}")

    if args.export_calibration:
        print(f"  export-width: {args.export_width}")
        print(f"  export-height: {args.export_height}")
        print(f"  export-fps: {args.export_fps}")

    main(
        args.video,
        args.video_hd,
        args.midi_mapping,
        args.debug,
        args.export_calibration,
        (args.export_width, args.export_height),
        args.export_fps,
        args.skip_calibration,
    )
