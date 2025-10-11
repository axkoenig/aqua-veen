import argparse
import signal
import time

import cv2
import mido
import numpy as np
import pandas as pd

# Constants
MIDI_RANGE = 127
BRIGHTNESS_RANGE = 255
NUM_CHANNELS = 4
DEBUG = False
BPM = 80

# MIDI Clock Constants
MIDI_CLOCKS_PER_BEAT = 24  # Standard MIDI clock resolution
BEATS_PER_BAR = 4

# Color mapping for channels (BGR format for OpenCV)
CHANNEL_COLORS = {
    0: (0, 0, 255),    # Red - Channel 1
    1: (0, 255, 0),    # Green - Channel 2
    2: (255, 0, 0),    # Blue - Channel 3
    3: (0, 255, 255),  # Yellow - Channel 4 (Kick)
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


class DigitoneChannel:
    """Base class for Digitone channel controllers"""
    
    def __init__(self, channel_num, midi_out, midi_mapping):
        """
        Initialize a Digitone channel.
        
        Args:
            channel_num: MIDI channel number (0-3 for Digitone channels 1-4)
            midi_out: MIDI output object
            midi_mapping: MIDI mapping DataFrame
        """
        self.channel_num = channel_num
        self.midi_out = midi_out
        self.midi_mapping = midi_mapping
        self.beat_count = 0
        
    def update(self, frame):
        """
        Update channel state based on current frame.
        Called every frame.
        
        Args:
            frame: Grayscale video frame
        """
        pass
    
    def on_beat(self, beat_number):
        """
        Called on each beat.
        
        Args:
            beat_number: Current beat number (1-indexed)
        """
        self.beat_count = beat_number


class Channel4Kick(DigitoneChannel):
    """
    Channel 4: Kick drum controller
    - Triggers on every 4th beat (standard kick pattern)
    - Loudness (velocity) controlled by average brightness
    """
    
    KICK_NOTE = 36  # Standard MIDI kick drum note
    
    def __init__(self, channel_num, midi_out, midi_mapping):
        super().__init__(channel_num, midi_out, midi_mapping)
        self.current_brightness = 0
        self.smoothing_factor = 0.3
        
    def update(self, frame):
        """Update brightness measurement from frame"""
        # Calculate average brightness with smoothing
        avg_brightness = np.mean(frame)
        self.current_brightness = (
            self.current_brightness * (1 - self.smoothing_factor) + 
            avg_brightness * self.smoothing_factor
        )
        debug_print(f"Channel 4 brightness: {self.current_brightness:.2f}")
    
    def on_beat(self, beat_number):
        """Trigger kick on every 4th beat"""
        super().on_beat(beat_number)
        
        self._trigger_kick()
    
    def _trigger_kick(self):
        """Send kick note with velocity based on brightness"""
        if not self.midi_out:
            return
        
        # Map brightness (0-255) to velocity (40-127)
        # Brighter = louder
        min_velocity = 0
        max_velocity = 127  
        velocity = int(min_velocity + (self.current_brightness / BRIGHTNESS_RANGE) * (max_velocity - min_velocity))
        velocity = max(min_velocity, min(max_velocity, velocity))
        
        # Send note on
        self.midi_out.send(
            mido.Message(
                'note_on',
                note=self.KICK_NOTE,
                velocity=velocity,
                channel=self.channel_num
            )
        )
        debug_print(f"Channel 4 KICK: velocity={velocity}, brightness={self.current_brightness:.2f}")
        
        # Send note off (short kick)
        # Note: In a real-time system, you might want to delay this slightly
        self.midi_out.send(
            mido.Message(
                'note_off',
                note=self.KICK_NOTE,
                velocity=0,
                channel=self.channel_num
            )
        )


def send_midi_control_change(
    midi_out,
    mapping,
    parameter_name,
    value,
    section="Filter parameters",
    channel=0,
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


class PlaceholderChannel(DigitoneChannel):
    """Placeholder for channels 1-3 (to be implemented)"""
    
    def update(self, frame):
        """No-op for now"""
        pass
    
    def on_beat(self, beat_number):
        """No-op for now"""
        super().on_beat(beat_number)




def send_stop_message(midi_out):
    """Send note-off to all MIDI channels."""
    if midi_out:
        for channel in range(16):
            for note in range(128):
                midi_out.send(mido.Message("note_off", note=note, channel=channel))


def main(
    midi_mapping_path="midi_mapping_digitone.csv",
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
        cv2.namedWindow("Digitone Control", cv2.WINDOW_NORMAL)

        # Initialize channel objects
        channels = [
            PlaceholderChannel(0, midi_out, midi_mapping),  # Channel 1
            PlaceholderChannel(1, midi_out, midi_mapping),  # Channel 2
            PlaceholderChannel(2, midi_out, midi_mapping),  # Channel 3
            Channel4Kick(3, midi_out, midi_mapping),        # Channel 4 (Kick)
        ]
        
        frame_count = 0

        print("Starting main loop...")
        print("Channel 4 (Kick): Triggers every 4th beat, loudness controlled by brightness")
        
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

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Update all channels with current frame
            for channel in channels:
                channel.update(gray_frame)

            # Trigger channel beat events
            if is_beat:
                for channel in channels:
                    channel.on_beat(midi_clock.beat_count)

            # Visualize
            vis_frame = frame.copy()
            
            # Display channel status
            y_offset = 30
            for i, channel in enumerate(channels):
                channel_num = i + 1
                color = CHANNEL_COLORS.get(i, (255, 255, 255))
                
                if isinstance(channel, Channel4Kick):
                    status_text = f"CH{channel_num} (Kick): Brightness={channel.current_brightness:.1f}"
                else:
                    status_text = f"CH{channel_num}: (placeholder)"
                
                cv2.putText(
                    vis_frame, status_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
                )
                cv2.putText(
                    vis_frame, status_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA
                )
                y_offset += 30

            # Draw beat indicator
            if is_beat:
                beat_text = f"BEAT {midi_clock.beat_count}"
                bar_num = (midi_clock.beat_count - 1) // BEATS_PER_BAR + 1
                beat_in_bar = (midi_clock.beat_count - 1) % BEATS_PER_BAR + 1
                
                cv2.putText(
                    vis_frame, beat_text, (10, vis_frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA
                )
                
                bar_text = f"Bar {bar_num}, Beat {beat_in_bar}"
                cv2.putText(
                    vis_frame, bar_text, (10, vis_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                )

            cv2.imshow("Digitone Control", vis_frame)

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
        default="midi_mapping_digitone.csv",
        help="Path to MIDI mapping CSV file",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    print("Command line arguments:")
    print(f"  midi-mapping: {args.midi_mapping}")
    print(f"  debug: {args.debug}")

    main(args.midi_mapping, args.debug)
