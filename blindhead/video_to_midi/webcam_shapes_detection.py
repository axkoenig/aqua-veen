"""
Same as blindspot_leipzig_piece_2.py, but uses webcam instead of video file.
"""

import argparse
import math
import signal
import time

import cv2
import mido
import numpy as np
import pandas as pd

# Constants
MIDI_RANGE = 127
BRIGHTNESS_RANGE = 255
NUM_SHAPES = 4
MAX_REVERB_DELAY = 60
DEBUG = False
MIDI_CHANNEL = 0
BPM = 80

# MIDI Clock Constants
MIDI_CLOCKS_PER_BEAT = 24  # Standard MIDI clock resolution
BEATS_PER_BAR = 4

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


class MIDIClock:
    """Handles MIDI clock generation at a specific BPM"""
    
    def __init__(self, midi_out, bpm=80):
        self.midi_out = midi_out
        self.bpm = bpm
        self.clock_interval = 60.0 / (bpm * MIDI_CLOCKS_PER_BEAT)  # Time between clock messages
        self.beat_interval = 60.0 / bpm  # Time between beats
        self.last_clock_time = time.time()
        self.last_beat_time = time.time()
        self.clock_count = 0
        self.beat_count = 0
        self.is_on_beat = False
        
    def start(self):
        """Send MIDI start message"""
        if self.midi_out:
            self.midi_out.send(mido.Message('start'))
            print(f"MIDI Clock started at {self.bpm} BPM")
    
    def stop(self):
        """Send MIDI stop message"""
        if self.midi_out:
            self.midi_out.send(mido.Message('stop'))
            print("MIDI Clock stopped")
    
    def update(self):
        """Update clock and check if we're on a beat. Call this frequently in main loop."""
        if not self.midi_out:
            return False
            
        current_time = time.time()
        
        # Send MIDI clock messages
        while current_time - self.last_clock_time >= self.clock_interval:
            self.midi_out.send(mido.Message('clock'))
            self.last_clock_time += self.clock_interval
            self.clock_count += 1
            
            # Check if we're on a beat (every 24 clocks)
            if self.clock_count % MIDI_CLOCKS_PER_BEAT == 0:
                self.beat_count += 1
                self.is_on_beat = True
                debug_print(f"Beat {self.beat_count} (Bar {(self.beat_count-1) // BEATS_PER_BAR + 1}, Beat {(self.beat_count-1) % BEATS_PER_BAR + 1})")
                return True
        
        self.is_on_beat = False
        return False


class AmbientEffectsState:
    def __init__(self):
        self.current_effect_value = MAX_REVERB_DELAY / 2
        self.smoothing_factor = 1


ambient_effects_state = AmbientEffectsState()


def send_midi_control_change(
    midi_out,
    mapping,
    parameter_name,
    value,
    section="Filter parameters",
    channel=MIDI_CHANNEL,
):
    """Sends a MIDI control change message based on a CSV mapping."""
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
    """Calculate average brightness and map to delay and reverb sends."""
    if midi_out is None:
        return

    global ambient_effects_state

    avg_brightness = np.mean(frame)
    target_effect_value = int(
        MAX_REVERB_DELAY * (1 - avg_brightness / BRIGHTNESS_RANGE)
    )

    alpha = ambient_effects_state.smoothing_factor
    ambient_effects_state.current_effect_value = (
        ambient_effects_state.current_effect_value * (1 - alpha)
        + target_effect_value * alpha
    )
    smoothed_effect_value = int(ambient_effects_state.current_effect_value)

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


class Shape:
    """
    Represents a detected bright region that can play MIDI notes.
    Notes are only triggered on beats.
    """

    # C# minor scale MIDI notes (spanning 3 octaves)
    SCALE_NOTES = [
        61, 63, 64, 66, 68, 69, 71,  # First octave
        73, 75, 76, 78, 80, 81, 83,  # Second octave
        85, 87, 88, 90, 92, 93, 95,  # Third octave
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
        """Initialize a new shape."""
        self.shape_counter = shape_counter
        self.centroid = centroid
        self.area = area
        self.frame_width, self.frame_height = frame_dims
        self.midi_out = midi_out
        self.midi_mapping = midi_mapping
        self.channel = channel
        self.missing_frames = 0
        self.max_missing_frames = 30
        self.is_active = True
        self.contour = None
        self.current_filter_value = 64
        self.filter_smoothing_factor = 0.1
        
        # Note state
        self.notes = []
        self.notes_playing = False
        self.pending_note_on = False  # Flag to trigger notes on next beat

        # Determine number of notes based on area
        self.frame_area = self.frame_width * self.frame_height
        self.normalized_area = self.area / self.frame_area
        self.num_notes = self.map_shape_area_to_num_notes()

        # Select chord notes
        self.notes = self._select_chord_notes()

        # Calculate initial velocity
        self.velocity = self._calculate_velocity()

        # Mark that we want to play notes on next beat
        self.pending_note_on = True
        debug_print(
            f"Created shape at {centroid} with notes {self.notes} (will play on next beat) on channel {self.channel}"
        )

    def _select_chord_notes(self):
        """Select a chord of appropriate size from the scale."""
        if len(self.SCALE_NOTES) < 7:
            print(f"Error: SCALE_NOTES must have at least 7 notes")
            return [self.SCALE_NOTES[0]] * min(self.num_notes, len(self.SCALE_NOTES))
        
        max_start_idx = len(self.SCALE_NOTES) - 7
        start_idx = np.random.randint(0, max_start_idx + 1)

        if self.num_notes == 1:
            return [self.SCALE_NOTES[start_idx]]
        elif self.num_notes == 2:
            return [self.SCALE_NOTES[start_idx], self.SCALE_NOTES[start_idx + 2]]
        elif self.num_notes == 3:
            return [self.SCALE_NOTES[start_idx], self.SCALE_NOTES[start_idx + 2], self.SCALE_NOTES[start_idx + 4]]
        else:
            return [
                self.SCALE_NOTES[start_idx],
                self.SCALE_NOTES[start_idx + 2],
                self.SCALE_NOTES[start_idx + 4],
                self.SCALE_NOTES[start_idx + 6],
            ]

    def _calculate_velocity(self):
        """Calculate MIDI velocity (30-127) based on shape's area."""
        velocity = int(30 + self.normalized_area * (127 - 30))
        return max(30, min(127, velocity))

    def _send_note_on(self):
        """Send MIDI note-on messages for all notes in the chord."""
        if self.midi_out and not self.notes_playing:
            for note in self.notes:
                self.midi_out.send(
                    mido.Message(
                        "note_on",
                        note=note,
                        velocity=self.velocity,
                        channel=self.channel,
                    )
                )
            self.notes_playing = True
            self.pending_note_on = False
            debug_print(f"Notes ON: {self.notes} on channel {self.channel}")

    def _send_note_off(self):
        """Send MIDI note-off messages for all notes."""
        if self.midi_out and self.notes_playing:
            for note in self.notes:
                self.midi_out.send(
                    mido.Message(
                        "note_off", note=note, velocity=0, channel=self.channel
                    )
                )
            self.notes_playing = False
            debug_print(f"Notes OFF: {self.notes} on channel {self.channel}")

    def map_shape_area_to_num_notes(self):
        """Map shape area to number of notes."""
        if self.normalized_area > 0.5:
            return 3
        elif self.normalized_area > 0.25:
            return 2
        else:
            return 1

    def update(self, centroid, area, frame, contour):
        """Update shape properties and adjust MIDI parameters."""
        if frame is None or frame.size == 0:
            debug_print("Warning: Invalid frame in Shape.update()")
            return
        
        self.centroid = centroid
        self.area = area
        self.normalized_area = self.area / self.frame_area
        self.missing_frames = 0
        self.contour = contour
        
        # Check if chord size changed
        new_num_notes = self.map_shape_area_to_num_notes()
        if new_num_notes != self.num_notes:
            # Stop current notes
            self._send_note_off()
            
            # Update chord
            self.num_notes = new_num_notes
            self.notes = self._select_chord_notes()
            self.pending_note_on = True  # Will play on next beat
            debug_print(
                f"Updated shape to play notes {self.notes} (will trigger on next beat)"
            )

        # Update filter based on brightness
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        mean_brightness = cv2.mean(frame, mask=mask)[0]

        filter_cut_off_start = 40
        target_filter_cutoff = int(mean_brightness * 127 / 255) + filter_cut_off_start

        alpha = self.filter_smoothing_factor
        self.current_filter_value = (
            self.current_filter_value * (1 - alpha) + target_filter_cutoff * alpha
        )
        smoothed_filter_value = int(self.current_filter_value)

        if self.midi_out and self.midi_mapping is not None:
            send_midi_control_change(
                self.midi_out,
                self.midi_mapping,
                "Filter 1: Frequency",
                smoothed_filter_value,
                section="Filters",
                channel=self.channel,
            )

    def on_beat(self):
        """Called when a beat occurs. Trigger notes if pending."""
        if self.pending_note_on:
            self._send_note_on()

    def destroy(self):
        """Stop playing all notes and clean up."""
        if self.is_active:
            self._send_note_off()
            self.is_active = False
        debug_print(
            f"Destroyed shape with notes {self.notes} on channel {self.channel}"
        )


def process_frame(midi_out, frame, active_shapes, midi_mapping, max_shapes=None):
    """Process frame and manage shape lifecycle."""
    global SHAPE_COUNTERS
    if max_shapes is None:
        max_shapes = NUM_SHAPES
    
    height, width = frame.shape[:2]
    frame_dims = (width, height)

    min_area = (width / 50) * (height / 50)

    # Detect regions
    _, thresh = cv2.threshold(frame, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Increment missing frames counter for all shapes
    for shape in active_shapes:
        shape.missing_frames += 1

    # Filter and sort contours
    valid_contours = []
    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[
            :max_shapes
        ]

    if not valid_contours:
        new_active_shapes = []
        for shape in active_shapes:
            if shape.missing_frames <= shape.max_missing_frames:
                new_active_shapes.append(shape)
            else:
                shape.destroy()
        return new_active_shapes

    # Calculate centroids
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
            best_match.update((cx, cy), area, frame, contour)
            new_active_shapes.append(best_match)
            used_current.add(i)
            used_active.add(best_idx)

    # Keep unmatched shapes if within tolerance
    for i, shape in enumerate(active_shapes):
        if i not in used_active:
            if shape.missing_frames <= shape.max_missing_frames:
                new_active_shapes.append(shape)
            else:
                shape.destroy()

    # Create new shapes for unmatched detections
    if len(new_active_shapes) < max_shapes:
        used_channels = {shape.channel for shape in new_active_shapes}

        for i, (cx, cy, area, contour) in enumerate(current_centroids):
            if i not in used_current:
                channel = 0
                while channel < max_shapes and channel in used_channels:
                    channel += 1

                SHAPE_COUNTERS[channel] += 1
                new_shape = Shape(
                    (cx, cy), area, frame_dims, midi_out, midi_mapping, 
                    channel=channel, shape_counter=SHAPE_COUNTERS[channel]
                )
                new_shape.contour = contour
                new_active_shapes.append(new_shape)
                used_channels.add(channel)

                if len(new_active_shapes) >= max_shapes:
                    break

    return new_active_shapes


def send_stop_message(midi_out):
    """Send note-off to all MIDI channels."""
    if midi_out:
        for channel in range(16):
            for note in range(128):
                midi_out.send(mido.Message("note_off", note=note, channel=channel))


def main(
    midi_mapping_path="midi_mapping_analog_four_mk2.csv",
    debug=False,
):
    global DEBUG
    DEBUG = debug

    print("Starting webcam MIDI program...")
    print(f"BPM: {BPM}")
    print(f"MIDI mapping: {midi_mapping_path}")

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

        # Open webcam
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to open webcam.")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default if webcam doesn't report FPS
        print(f"Webcam FPS: {fps}")

        # Initialize MIDI clock
        midi_clock = MIDIClock(midi_out, BPM)
        midi_clock.start()

        def signal_handler(sig, frame):
            print("Ctrl+C pressed, exiting...")
            midi_clock.stop()
            send_stop_message(midi_out)
            cap.release()
            cv2.destroyAllWindows()
            if midi_out:
                midi_out.close()
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Create visualization window
        cv2.namedWindow("Shapes Detection", cv2.WINDOW_NORMAL)

        active_shapes = []
        frame_count = 0

        print("Starting main loop...")
        while True:
            frame_count += 1

            # Update MIDI clock and check for beats
            is_beat = midi_clock.update()

            # Read webcam frame
            ret, frame = cap.read()
            if not ret:
                print("Warning: Couldn't read webcam frame")
                time.sleep(0.01)
                continue

            # Process frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            active_shapes = process_frame(
                midi_out, gray_frame, active_shapes, midi_mapping, max_shapes=NUM_SHAPES
            )

            # Trigger notes on beat
            if is_beat:
                for shape in active_shapes:
                    shape.on_beat()

            # Apply ambient effects
            send_ambient_effects(
                midi_out, gray_frame, midi_mapping, channel=MIDI_CHANNEL
            )

            # Visualize shapes
            vis_frame = frame.copy()
            for shape in active_shapes:
                if shape.is_active:
                    color = CHANNEL_COLORS.get(shape.channel, (255, 255, 255))
                    
                    # Draw contour with transparency
                    if shape.contour is not None:
                        overlay = vis_frame.copy()
                        cv2.drawContours(overlay, [shape.contour], -1, color, thickness=-1)
                        alpha = 0.3
                        vis_frame = cv2.addWeighted(vis_frame, 1 - alpha, overlay, alpha, 0)
                    
                    # Draw centroid marker
                    centroid = shape.centroid
                    cv2.circle(
                        vis_frame, 
                        (int(centroid[0]), int(centroid[1])), 
                        20, 
                        color, 
                        -1
                    )
                    
                    # Draw counter text
                    counter_text = str(shape.shape_counter)
                    text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_x = int(centroid[0] - text_size[0] // 2)
                    text_y = int(centroid[1] + text_size[1] // 2)
                    
                    cv2.putText(
                        vis_frame, counter_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        vis_frame, counter_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
                    )
                    
                    # Draw num_notes
                    notes_text = f"notes={shape.num_notes}"
                    notes_text_size = cv2.getTextSize(notes_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    notes_text_x = int(centroid[0] - notes_text_size[0] // 2)
                    notes_text_y = int(centroid[1] + text_size[1] + 15)
                    
                    cv2.putText(
                        vis_frame, notes_text, (notes_text_x, notes_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        vis_frame, notes_text, (notes_text_x, notes_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                    )

            # Draw beat indicator
            if is_beat:
                beat_indicator = "BEAT"
                cv2.putText(
                    vis_frame, beat_indicator, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA
                )

            cv2.imshow("Shapes Detection", vis_frame)

            # Check for ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("ESC pressed, exiting...")
                break

    except Exception as e:
        import traceback
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())

    finally:
        print("Cleaning up resources...")
        if 'midi_clock' in locals():
            midi_clock.stop()
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        if 'midi_out' in locals() and midi_out:
            midi_out.close()
        print("Program finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process webcam feed to MIDI with clock sync")
    parser.add_argument(
        "--midi-mapping",
        type=str,
        default="midi_mapping_analog_four_mk2.csv",
        help="Path to MIDI mapping CSV file",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    print("Command line arguments:")
    print(f"  midi-mapping: {args.midi_mapping}")
    print(f"  debug: {args.debug}")

    main(args.midi_mapping, args.debug)
