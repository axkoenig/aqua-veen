#!/usr/bin/env python3
"""
Combined script: Generate MIDI from webcam AND stream to HackTV
- Uses single webcam feed for both MIDI generation and TV transmission
- MIDI generation runs independently in main thread
- HackTV video mixing runs in subprocess pipeline
- Both can be enabled/disabled via command line flags
"""

import argparse
import signal
import time
import subprocess
import threading
import sys
from pathlib import Path

import cv2
import mido
import numpy as np
import pandas as pd

try:
    from pythonosc import dispatcher, osc_server
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False

# Constants
MIDI_RANGE = 127
BRIGHTNESS_RANGE = 255
NUM_CHANNELS = 4
DEBUG = False
BPM = 80

# MIDI Clock Constants
MIDI_CLOCKS_PER_BEAT = 24
BEATS_PER_BAR = 4

# Channel min/max thresholds (CC values 0-127)
# Channel 1: Red → Track 1 SYN 1 Feedback
CH1_MIN = 0
CH1_MAX = 127

# Channel 2: Green → Track 2 Amp Drive
CH2_MIN = 50
CH2_MAX = 100

# Channel 3: Blue → Track 2 Filter Frequency
CH3_MIN = 50
CH3_MAX = 100

# Channel 4: Total Brightness → Track 4 Volume
CH4_MIN = 0
CH4_MAX = 127

# HackTV settings
VIDEO_LOOP_FILE = "/home/weast/org/projects/Art/sindelfingen/Teaser05.mp4"
VIDEO_RESOLUTION = "720x576"  # PAL resolution
VIDEO_FRAMERATE = 25
HACKTV_MODE = "g"
HACKTV_FREQ = "471250000"
HACKTV_SAMPLE_RATE = "16000000"
HACKTV_GAIN = "47"

# OSC settings
OSC_IP = "0.0.0.0"
OSC_PORT = 8000

# Color mapping for channels (BGR format for OpenCV)
CHANNEL_COLORS = {
    0: (0, 0, 255),    # Red - Channel 1
    1: (0, 255, 0),    # Green - Channel 2
    2: (255, 0, 0),    # Blue - Channel 3
    3: (0, 255, 255),  # Yellow - Channel 4 (Kick)
}

# Global state
hacktv_processes = []
osc_server_instance = None
shutdown_requested = threading.Event()
restart_hacktv_requested = threading.Event()

# Frame queue for shared webcam mode
import queue
frame_queue = queue.Queue(maxsize=5)  # Limit queue size to prevent memory buildup
ffmpeg_stdin = None  # Will hold the FFmpeg stdin pipe


def debug_print(*args, **kwargs):
    """Print only when DEBUG is True"""
    if DEBUG:
        print(*args, **kwargs)


# ============================================================================
# MIDI Components
# ============================================================================

def init_midi_output(device_index=None):
    """Initialize MIDI output.

    Args:
        device_index: Optional index of MIDI device to use. If None, uses first device.
    """
    try:
        output_names = mido.get_output_names()

        if not output_names:
            print("No MIDI output available.")
            return None

        print("\n" + "="*60)
        print("Available MIDI Output Devices:")
        print("="*60)
        for i, name in enumerate(output_names):
            print(f"  [{i}] {name}")
        print("="*60)

        # Use specified device or default to first one
        if device_index is not None and 0 <= device_index < len(output_names):
            selected_device = output_names[device_index]
        else:
            selected_device = output_names[0]
            device_index = 0

        midi_out = mido.open_output(selected_device)
        print(f"\n✓ Using MIDI device [{device_index}]: {selected_device}")
        print("="*60 + "\n")
        return midi_out
    except Exception as e:
        print(f"Error initializing MIDI: {e}")
        return None


class MIDIClock:
    """Handles MIDI clock generation at a specific BPM"""

    def __init__(self, midi_out, bpm=80):
        self.midi_out = midi_out
        self.bpm = bpm
        self.clock_interval = 60.0 / (bpm * MIDI_CLOCKS_PER_BEAT)
        self.beat_interval = 60.0 / bpm
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
        """Update clock and check if we're on a beat."""
        if not self.midi_out:
            return False

        current_time = time.time()

        while current_time - self.last_clock_time >= self.clock_interval:
            self.midi_out.send(mido.Message('clock'))
            self.last_clock_time += self.clock_interval
            self.clock_count += 1

            if self.clock_count % MIDI_CLOCKS_PER_BEAT == 0:
                self.beat_count += 1
                self.is_on_beat = True
                debug_print(f"Beat {self.beat_count}")
                return True

        self.is_on_beat = False
        return False


class DigitoneChannel:
    """Base class for Digitone channel controllers"""

    def __init__(self, channel_num, midi_out, midi_mapping):
        self.channel_num = channel_num
        self.midi_out = midi_out
        self.midi_mapping = midi_mapping
        self.beat_count = 0

    def update(self, frame):
        """Update channel state based on current frame."""
        pass

    def on_beat(self, beat_number):
        """Called on each beat."""
        self.beat_count = beat_number


class Channel4Brightness(DigitoneChannel):
    """Channel 4: Total brightness → CC control"""

    def __init__(self, channel_num, midi_out, midi_mapping):
        super().__init__(channel_num, midi_out, midi_mapping)
        self.current_brightness = 0
        self.smoothing_factor = 0.3
        self.cc_number = 7  # Volume (could be changed to any CC)
        self.last_sent_value = -1

    def update(self, frame):
        """Update brightness measurement from grayscale frame and send CC"""
        avg_brightness = np.mean(frame)
        self.current_brightness = (
            self.current_brightness * (1 - self.smoothing_factor) +
            avg_brightness * self.smoothing_factor
        )
        debug_print(f"Channel 4 (Total brightness) brightness: {self.current_brightness:.2f}")

        # Send CC immediately
        self._send_cc()

    def _send_cc(self):
        """Send CC message based on brightness"""
        if not self.midi_out:
            return

        # Map brightness (0-255) to channel's min-max range
        normalized = self.current_brightness / BRIGHTNESS_RANGE
        cc_value = int(CH4_MIN + normalized * (CH4_MAX - CH4_MIN))
        cc_value = max(CH4_MIN, min(CH4_MAX, cc_value))

        # Only send if value changed
        if cc_value != self.last_sent_value:
            self.midi_out.send(
                mido.Message(
                    'control_change',
                    control=self.cc_number,
                    value=cc_value,
                    channel=self.channel_num
                )
            )
            self.last_sent_value = cc_value
            debug_print(f"Channel 4 (Total brightness) CC{self.cc_number}: {cc_value}")


class Channel1Red(DigitoneChannel):
    """Channel 1: Red color brightness → SYN 1 Feedback"""

    def __init__(self, channel_num, midi_out, midi_mapping):
        super().__init__(channel_num, midi_out, midi_mapping)
        self.current_brightness = 0
        self.smoothing_factor = 0.3
        self.cc_number = 19  # SYN 1 Feedback
        self.last_sent_value = -1  # Track last sent value to avoid spamming

    def update_from_bgr(self, bgr_frame):
        """Update from BGR frame to get red channel and send CC"""
        # Extract red channel (index 2 in BGR)
        red_channel = bgr_frame[:, :, 2]
        avg_brightness = np.mean(red_channel)

        self.current_brightness = (
            self.current_brightness * (1 - self.smoothing_factor) +
            avg_brightness * self.smoothing_factor
        )
        debug_print(f"Channel 1 (Red) brightness: {self.current_brightness:.2f}")

        # Send CC immediately
        self._send_cc()

    def _send_cc(self):
        """Send CC message based on brightness"""
        if not self.midi_out:
            return

        # Map brightness (0-255) to channel's min-max range
        normalized = self.current_brightness / BRIGHTNESS_RANGE
        cc_value = int(CH1_MIN + normalized * (CH1_MAX - CH1_MIN))
        cc_value = max(CH1_MIN, min(CH1_MAX, cc_value))

        # Only send if value changed
        if cc_value != self.last_sent_value:
            self.midi_out.send(
                mido.Message(
                    'control_change',
                    control=self.cc_number,
                    value=cc_value,
                    channel=self.channel_num
                )
            )
            self.last_sent_value = cc_value
            debug_print(f"Channel 1 (Red) → Track 1 Feedback CC{self.cc_number}: {cc_value}")


class Channel2Green(DigitoneChannel):
    """Channel 2: Green color brightness → Amp Drive"""

    def __init__(self, channel_num, midi_out, midi_mapping):
        super().__init__(channel_num, midi_out, midi_mapping)
        self.current_brightness = 0
        self.smoothing_factor = 0.3
        self.cc_number = 9  # Amp Drive
        self.last_sent_value = -1

    def update_from_bgr(self, bgr_frame):
        """Update from BGR frame to get green channel and send CC"""
        # Extract green channel (index 1 in BGR)
        green_channel = bgr_frame[:, :, 1]
        avg_brightness = np.mean(green_channel)

        self.current_brightness = (
            self.current_brightness * (1 - self.smoothing_factor) +
            avg_brightness * self.smoothing_factor
        )
        debug_print(f"Channel 2 (Green) brightness: {self.current_brightness:.2f}")

        # Send CC immediately
        self._send_cc()

    def _send_cc(self):
        """Send CC message based on brightness"""
        if not self.midi_out:
            return

        # Map brightness (0-255) to channel's min-max range
        normalized = self.current_brightness / BRIGHTNESS_RANGE
        cc_value = int(CH2_MIN + normalized * (CH2_MAX - CH2_MIN))
        cc_value = max(CH2_MIN, min(CH2_MAX, cc_value))

        # Only send if value changed
        if cc_value != self.last_sent_value:
            self.midi_out.send(
                mido.Message(
                    'control_change',
                    control=self.cc_number,
                    value=cc_value,
                    channel=self.channel_num
                )
            )
            self.last_sent_value = cc_value
            debug_print(f"Channel 2 (Green) CC{self.cc_number}: {cc_value}")


class Channel3Blue(DigitoneChannel):
    """Channel 3: Blue color brightness → Filter Frequency on Track 2"""

    def __init__(self, channel_num, midi_out, midi_mapping):
        super().__init__(channel_num, midi_out, midi_mapping)
        self.current_brightness = 0
        self.smoothing_factor = 0.3
        self.cc_number = 23  # Filter Frequency
        self.last_sent_value = -1
        # Override channel_num to send to track 2 (MIDI channel 2 = index 1)
        self.channel_num = 1

    def update_from_bgr(self, bgr_frame):
        """Update from BGR frame to get blue channel and send CC"""
        # Extract blue channel (index 0 in BGR)
        blue_channel = bgr_frame[:, :, 0]
        avg_brightness = np.mean(blue_channel)

        self.current_brightness = (
            self.current_brightness * (1 - self.smoothing_factor) +
            avg_brightness * self.smoothing_factor
        )
        debug_print(f"Channel 3 (Blue) brightness: {self.current_brightness:.2f}")

        # Send CC immediately
        self._send_cc()

    def _send_cc(self):
        """Send CC message based on brightness"""
        if not self.midi_out:
            return

        # Map brightness (0-255) to channel's min-max range
        normalized = self.current_brightness / BRIGHTNESS_RANGE
        cc_value = int(CH3_MIN + normalized * (CH3_MAX - CH3_MIN))
        cc_value = max(CH3_MIN, min(CH3_MAX, cc_value))

        # Only send if value changed
        if cc_value != self.last_sent_value:
            self.midi_out.send(
                mido.Message(
                    'control_change',
                    control=self.cc_number,
                    value=cc_value,
                    channel=self.channel_num
                )
            )
            self.last_sent_value = cc_value
            debug_print(f"Channel 3 (Blue) → Track 2 CC{self.cc_number}: {cc_value}")


def send_stop_message(midi_out):
    """Send note-off to all MIDI channels."""
    if midi_out:
        for channel in range(16):
            for note in range(128):
                midi_out.send(mido.Message("note_off", note=note, channel=channel))


# ============================================================================
# HackTV Video Mixing Components
# ============================================================================

class MixerState:
    def __init__(self):
        self.mix = 0.5
        self.lock = threading.Lock()

    def get_mix(self):
        with self.lock:
            return self.mix

    def set_mix(self, value):
        with self.lock:
            old_mix = self.mix
            self.mix = max(0.0, min(1.0, value))
            return self.mix, old_mix

mixer_state = MixerState()


def osc_mix_handler(address, *args):
    """Handle OSC messages for mix control"""
    if len(args) > 0:
        mix_value = float(args[0])
        new_mix, old_mix = mixer_state.set_mix(mix_value)
        if abs(new_mix - old_mix) > 0.01:
            restart_hacktv_requested.set()
        print(f"\rMix: {new_mix:.2f} - Restarting HackTV...  ", end="", flush=True)


def start_osc_server():
    """Start OSC server in a separate thread"""
    if not OSC_AVAILABLE:
        return

    global osc_server_instance

    disp = dispatcher.Dispatcher()
    disp.map("/mix", osc_mix_handler)

    try:
        osc_server_instance = osc_server.ThreadingOSCUDPServer(
            (OSC_IP, OSC_PORT), disp
        )
        print(f"OSC server listening on {OSC_IP}:{OSC_PORT} for /mix messages")

        server_thread = threading.Thread(target=osc_server_instance.serve_forever)
        server_thread.daemon = True
        server_thread.start()
    except Exception as e:
        print(f"Could not start OSC server: {e}")


def find_webcam():
    """Auto-detect and validate webcam device"""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            check=True
        )

        print("Available video devices:")
        print(result.stdout)

        devices = []
        for line in result.stdout.split('\n'):
            if '/dev/video' in line:
                device = line.strip()
                if device.endswith(('0', '2', '4', '6', '8')):
                    devices.append(device)

        for device in devices:
            print(f"\nTesting {device}...", end=" ")
            try:
                cap_result = subprocess.run(
                    ["v4l2-ctl", "-d", device, "--all"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )

                if "Video Capture" in cap_result.stdout:
                    print(f"✓ Valid capture device")
                    response = input(f"Use {device}? (y/n/skip): ").lower()
                    if response == 'y':
                        return device
                    elif response == 's' or response == 'skip':
                        continue
                else:
                    print("✗ Not a capture device")

            except subprocess.TimeoutExpired:
                print("✗ Timeout")
            except Exception as e:
                print(f"✗ Error: {e}")

        device = input("\nEnter video device manually (e.g., /dev/video0): ").strip()
        if device:
            return device

    except subprocess.CalledProcessError:
        print("Could not list devices with v4l2-ctl")
    except FileNotFoundError:
        print("v4l2-ctl not found. Please install v4l-utils")

    print("\nTrying common video devices...")
    for device in ["/dev/video0", "/dev/video2", "/dev/video4"]:
        if Path(device).exists():
            print(f"Found {device}")
            response = input(f"Use {device}? (y/n): ").lower()
            if response == 'y':
                return device

    return None


def build_ffmpeg_command(webcam_device, use_stdin=False, frame_width=640, frame_height=480):
    """Build FFmpeg command with current mix value

    Args:
        webcam_device: Path to webcam device (only used if use_stdin=False)
        use_stdin: If True, accept rawvideo from stdin instead of v4l2
        frame_width: Width of input frames (when use_stdin=True)
        frame_height: Height of input frames (when use_stdin=True)
    """
    mix = mixer_state.get_mix()
    width, height = VIDEO_RESOLUTION.split('x')

    if use_stdin:
        # Accept BGR24 rawvideo from stdin (OpenCV format)
        cmd = [
            "ffmpeg",
            "-loglevel", "info",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{frame_width}x{frame_height}",
            "-framerate", "25",
            "-i", "-",  # stdin
            "-stream_loop", "-1",
            "-i", VIDEO_LOOP_FILE,
            "-filter_complex",
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[webcam];"
            f"[1:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[loop];"
            f"[webcam][loop]blend=all_expr='A*(1-{mix})+B*{mix}':shortest=1[out]",
            "-map", "[out]",
            "-map", "1:a",
            "-c:v", "mpeg2video",
            "-r", str(VIDEO_FRAMERATE),
            "-b:v", "4M",
            "-maxrate", "4M",
            "-bufsize", "8M",
            "-c:a", "mp2",
            "-b:a", "192k",
            "-f", "mpegts",
            "-"
        ]
    else:
        # Original v4l2 capture mode
        cmd = [
            "ffmpeg",
            "-loglevel", "info",
            "-f", "v4l2",
            "-input_format", "mjpeg",
            "-framerate", "25",
            "-video_size", "640x480",
            "-i", webcam_device,
            "-stream_loop", "-1",
            "-i", VIDEO_LOOP_FILE,
            "-filter_complex",
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[webcam];"
            f"[1:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[loop];"
            f"[webcam][loop]blend=all_expr='A*(1-{mix})+B*{mix}':shortest=1[out]",
            "-map", "[out]",
            "-map", "1:a",
            "-c:v", "mpeg2video",
            "-r", str(VIDEO_FRAMERATE),
            "-b:v", "4M",
            "-maxrate", "4M",
            "-bufsize", "8M",
            "-c:a", "mp2",
            "-b:a", "192k",
            "-f", "mpegts",
            "-"
        ]

    print("\n" + "=" * 60)
    print("DEBUG: FFmpeg Command")
    print("=" * 60)
    if use_stdin:
        print(f"Webcam source: OpenCV frames via stdin")
        print(f"Frame size: {frame_width}x{frame_height}")
    else:
        print(f"Webcam device: {webcam_device}")
    print(f"Video loop file: {VIDEO_LOOP_FILE}")
    print(f"Mix value: {mix:.2f} (0=all webcam, 1=all loop)")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60 + "\n")

    return cmd


def stderr_monitor(process, name):
    """Monitor and print stderr from a process"""
    try:
        for line in iter(process.stderr.readline, b''):
            if line:
                decoded = line.decode('utf-8', errors='replace').strip()
                if decoded:
                    print(f"[{name}] {decoded}")
    except:
        pass


def frame_writer_thread(stdin_pipe):
    """Write frames from queue to FFmpeg stdin"""
    global ffmpeg_stdin
    ffmpeg_stdin = stdin_pipe

    print("DEBUG: Frame writer thread started")
    frames_written = 0

    try:
        while not shutdown_requested.is_set():
            try:
                # Get frame from queue with timeout
                frame = frame_queue.get(timeout=0.1)

                if frame is None:  # Sentinel value to stop
                    break

                # Write raw frame data to FFmpeg stdin
                stdin_pipe.write(frame.tobytes())
                frames_written += 1

                if frames_written % 100 == 0:
                    print(f"DEBUG: Wrote {frames_written} frames to FFmpeg")

            except queue.Empty:
                continue
            except BrokenPipeError:
                print("DEBUG: FFmpeg stdin pipe broken")
                break
            except Exception as e:
                print(f"DEBUG: Error writing frame: {e}")
                break

    except Exception as e:
        print(f"DEBUG: Frame writer thread error: {e}")
    finally:
        try:
            stdin_pipe.close()
        except:
            pass
        print(f"DEBUG: Frame writer thread stopped (wrote {frames_written} frames)")


def start_hacktv_pipeline(webcam_device, use_shared_webcam=False, frame_width=640, frame_height=480):
    """Start FFmpeg and HackTV pipeline

    Args:
        webcam_device: Path to webcam device (only used if use_shared_webcam=False)
        use_shared_webcam: If True, accept frames from OpenCV via stdin
        frame_width: Width of OpenCV frames (when use_shared_webcam=True)
        frame_height: Height of OpenCV frames (when use_shared_webcam=True)
    """
    global hacktv_processes

    ffmpeg_cmd = build_ffmpeg_command(
        webcam_device,
        use_stdin=use_shared_webcam,
        frame_width=frame_width,
        frame_height=frame_height
    )
    hacktv_cmd = [
        "hacktv",
        "-m", HACKTV_MODE,
        "-f", HACKTV_FREQ,
        "-s", HACKTV_SAMPLE_RATE,
        "-g", HACKTV_GAIN,
        "--repeat",
        "-"
    ]

    print("DEBUG: HackTV Command")
    print("=" * 60)
    print(f"Command: {' '.join(hacktv_cmd)}")
    print("=" * 60)

    print("\nDEBUG: Starting FFmpeg process...")
    if use_shared_webcam:
        # FFmpeg will read from stdin
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Start frame writer thread
        writer_thread = threading.Thread(
            target=frame_writer_thread,
            args=(ffmpeg_proc.stdin,),
            daemon=True
        )
        writer_thread.start()
    else:
        # FFmpeg will read from v4l2 device
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    print("DEBUG: Starting HackTV process...")
    hacktv_proc = subprocess.Popen(
        hacktv_cmd,
        stdin=ffmpeg_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    ffmpeg_proc.stdout.close()

    # Start stderr monitoring threads
    ffmpeg_monitor = threading.Thread(
        target=stderr_monitor,
        args=(ffmpeg_proc, "FFmpeg"),
        daemon=True
    )
    hacktv_monitor = threading.Thread(
        target=stderr_monitor,
        args=(hacktv_proc, "HackTV"),
        daemon=True
    )

    ffmpeg_monitor.start()
    hacktv_monitor.start()

    print("DEBUG: Pipeline started, monitoring output...")

    hacktv_processes = [ffmpeg_proc, hacktv_proc]
    return hacktv_processes


def stop_hacktv_pipeline():
    """Stop current HackTV pipeline"""
    global hacktv_processes, ffmpeg_stdin

    # Signal frame writer thread to stop by sending sentinel
    try:
        frame_queue.put(None, timeout=0.5)
    except:
        pass

    # Close FFmpeg stdin if it exists
    if ffmpeg_stdin:
        try:
            ffmpeg_stdin.close()
        except:
            pass
        ffmpeg_stdin = None

    # Terminate processes gracefully
    for proc in hacktv_processes:
        try:
            proc.terminate()
        except:
            pass

    time.sleep(0.2)  # Give processes time to terminate

    # Force kill if still running
    for proc in hacktv_processes:
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=1)  # Wait for kill to complete
        except:
            pass

    hacktv_processes = []


def hacktv_control_thread(webcam_device, use_shared_webcam=False):
    """Monitor and restart HackTV pipeline when needed

    Args:
        webcam_device: Path to webcam device
        use_shared_webcam: If True, using shared webcam mode
    """

    print("\nHackTV Keyboard controls:")
    print("  ← → : Adjust mix (±5%)")
    print("  1-9 : Set mix presets")
    print("  0   : 50/50 mix")
    print("  Q   : All webcam (0%)")
    print("  W   : All loop (100%)")
    print("  R   : Force restart pipeline")

    last_change_time = [0]

    def debounce_watcher():
        while not shutdown_requested.is_set():
            if last_change_time[0] > 0 and (time.time() - last_change_time[0]) >= 1.0:
                if not restart_hacktv_requested.is_set():
                    print(f"\rMix: {mixer_state.get_mix():.2f} - Applying...     ", end="", flush=True)
                    restart_hacktv_requested.set()
                last_change_time[0] = 0
            time.sleep(0.1)

    debounce_thread = threading.Thread(target=debounce_watcher)
    debounce_thread.daemon = True
    debounce_thread.start()

    # Keyboard control
    try:
        import tty
        import termios

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())

            while not shutdown_requested.is_set():
                if sys.stdin in []:
                    time.sleep(0.1)
                    continue

                char = sys.stdin.read(1)
                changed = False

                if char == 'q' or char == 'Q':
                    mixer_state.set_mix(0.0)
                    changed = True
                elif char == 'w' or char == 'W':
                    mixer_state.set_mix(1.0)
                    changed = True
                elif char == '0':
                    mixer_state.set_mix(0.5)
                    changed = True
                elif char.isdigit():
                    mixer_state.set_mix(int(char) / 10.0)
                    changed = True
                elif char == 'r' or char == 'R':
                    restart_hacktv_requested.set()
                elif char == '\x1b':
                    next1, next2 = sys.stdin.read(2)
                    if next1 == '[':
                        if next2 == 'C':  # Right arrow
                            current = mixer_state.get_mix()
                            mixer_state.set_mix(current + 0.05)
                            changed = True
                        elif next2 == 'D':  # Left arrow
                            current = mixer_state.get_mix()
                            mixer_state.set_mix(current - 0.05)
                            changed = True

                if changed:
                    last_change_time[0] = time.time()
                    print(f"\rMix: {mixer_state.get_mix():.2f} (pending...)     ", end="", flush=True)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    except:
        pass


def hacktv_manager_thread(webcam_device, use_shared_webcam=False, frame_width=640, frame_height=480):
    """Manage HackTV pipeline lifecycle

    Args:
        webcam_device: Path to webcam device
        use_shared_webcam: If True, using shared webcam mode
        frame_width: Width of OpenCV frames (when use_shared_webcam=True)
        frame_height: Height of OpenCV frames (when use_shared_webcam=True)
    """

    try:
        print(f"Starting HackTV with mix: {mixer_state.get_mix():.2f}")
        start_hacktv_pipeline(webcam_device, use_shared_webcam, frame_width, frame_height)
        print("✓ HackTV streaming active!")

        last_health_check = time.time()

        while not shutdown_requested.is_set():
            if restart_hacktv_requested.is_set():
                restart_hacktv_requested.clear()
                print(f"\n🔄 Restarting HackTV with mix: {mixer_state.get_mix():.2f}...")
                stop_hacktv_pipeline()
                time.sleep(0.05)
                start_hacktv_pipeline(webcam_device, use_shared_webcam, frame_width, frame_height)
                print("✓ HackTV streaming resumed")

            # Check if processes died
            for i, proc in enumerate(hacktv_processes):
                if proc.poll() is not None:
                    proc_name = "FFmpeg" if i == 0 else "HackTV"
                    exit_code = proc.returncode
                    print(f"\n⚠ {proc_name} process died with exit code {exit_code}!")
                    print("DEBUG: Restarting pipeline in 1 second...")
                    stop_hacktv_pipeline()
                    time.sleep(1)
                    start_hacktv_pipeline(webcam_device, use_shared_webcam, frame_width, frame_height)
                    break

            # Periodic health check (every 10 seconds)
            current_time = time.time()
            if current_time - last_health_check > 10:
                if len(hacktv_processes) == 2:
                    ffmpeg_status = "running" if hacktv_processes[0].poll() is None else "dead"
                    hacktv_status = "running" if hacktv_processes[1].poll() is None else "dead"
                    print(f"DEBUG: Health check - FFmpeg: {ffmpeg_status}, HackTV: {hacktv_status}")
                last_health_check = current_time

            time.sleep(0.2)

    except Exception as e:
        print(f"HackTV Manager Error: {e}")
    finally:
        stop_hacktv_pipeline()


# ============================================================================
# Main Program
# ============================================================================

def main(
    midi_mapping_path="midi_mapping_digitone.csv",
    debug=False,
    enable_midi=True,
    enable_hacktv=False,
    webcam_device=None,
    midi_device_index=None,
):
    global DEBUG
    DEBUG = debug

    print("=" * 60)
    print("WEBCAM → MIDI + HACKTV COMBINED")
    print("=" * 60)
    print(f"MIDI Generation: {'ENABLED' if enable_midi else 'DISABLED'}")
    print(f"HackTV Streaming: {'ENABLED' if enable_hacktv else 'DISABLED'}")
    print("=" * 60)

    # Initialize MIDI if enabled
    midi_out = None
    midi_mapping = None
    channels = None

    if enable_midi:
        midi_out = init_midi_output(midi_device_index)

        try:
            midi_mapping = pd.read_csv(midi_mapping_path)
            if DEBUG:
                debug_print(midi_mapping)
        except Exception as e:
            print(f"Error loading MIDI mapping: {e}")
            if enable_midi:
                return

        channels = [
            Channel1Red(0, midi_out, midi_mapping),
            Channel2Green(1, midi_out, midi_mapping),
            Channel3Blue(2, midi_out, midi_mapping),
            Channel4Brightness(3, midi_out, midi_mapping),
        ]

        print("MIDI channels initialized (CC only, no clock)")
        print(f"Channel 1 (Red) → Track 1: SYN 1 Feedback (CC 19) [{CH1_MIN}-{CH1_MAX}]")
        print(f"Channel 2 (Green) → Track 2: Amp Drive (CC 9) [{CH2_MIN}-{CH2_MAX}]")
        print(f"Channel 3 (Blue) → Track 2: Filter Frequency (CC 23) [{CH3_MIN}-{CH3_MAX}]")
        print(f"Channel 4 (Total Brightness) → Track 4: Volume (CC 7) [{CH4_MIN}-{CH4_MAX}]")

    # Setup webcam
    if webcam_device is None:
        print("\nDetecting webcam...")
        webcam_device = find_webcam()
        if webcam_device is None:
            print("\nError: No webcam device found!")
            return

    print(f"\n✓ Webcam: {webcam_device}")

    # Determine webcam sharing mode
    use_shared_webcam = enable_midi and enable_hacktv
    frame_width = 640
    frame_height = 480

    # Open webcam for MIDI (and HackTV if in shared mode)
    cap = None
    if enable_midi or use_shared_webcam:
        print("Opening webcam for MIDI generation...")
        if use_shared_webcam:
            print("(Webcam will be shared with HackTV)")
        cap = cv2.VideoCapture(webcam_device)
        if not cap.isOpened():
            print("Error: Unable to open webcam.")
            return

        # Get actual frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        print(f"Webcam FPS: {fps}")
        print(f"Webcam resolution: {frame_width}x{frame_height}")

    # Setup HackTV if enabled
    if enable_hacktv:
        video_path = Path(VIDEO_LOOP_FILE)
        if not video_path.exists():
            print(f"Warning: Video loop file '{VIDEO_LOOP_FILE}' not found!")
            print("HackTV will be disabled.")
            enable_hacktv = False
        else:
            # Get file size in MB
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"\n{'='*60}")
            print("DEBUG: Video File Status")
            print(f"{'='*60}")
            print(f"✓ Loop video: {VIDEO_LOOP_FILE}")
            print(f"✓ File exists: Yes")
            print(f"✓ File size: {file_size_mb:.2f} MB")
            print(f"✓ TV frequency: {HACKTV_FREQ} Hz (PAL {HACKTV_MODE.upper()})")
            print(f"✓ Initial mix: {mixer_state.get_mix():.2f}")
            if use_shared_webcam:
                print(f"✓ Shared webcam mode: ENABLED")
                print(f"✓ OpenCV will feed frames to FFmpeg via stdin")
            print(f"{'='*60}\n")

            if OSC_AVAILABLE:
                start_osc_server()

            # Start HackTV manager thread
            hacktv_thread = threading.Thread(
                target=hacktv_manager_thread,
                args=(webcam_device, use_shared_webcam, frame_width, frame_height)
            )
            hacktv_thread.daemon = True
            hacktv_thread.start()

            # Start HackTV control thread
            control_thread = threading.Thread(
                target=hacktv_control_thread,
                args=(webcam_device, use_shared_webcam)
            )
            control_thread.daemon = True
            control_thread.start()

    print("\nPress Ctrl+C to stop")
    if enable_midi:
        print("Press ESC in video window to exit")
    print()

    def signal_handler(sig, frame):
        print("\n\nStopping...")
        shutdown_requested.set()

        # Stop HackTV first (closes frame queue)
        print("Stopping HackTV pipeline...")
        stop_hacktv_pipeline()

        # Give threads time to stop
        time.sleep(0.3)

        # Stop MIDI (send all notes off)
        if midi_out:
            send_stop_message(midi_out)

        # Release camera
        if cap:
            try:
                cap.release()
            except:
                pass

        try:
            cv2.destroyAllWindows()
        except:
            pass

        # Stop OSC server
        if osc_server_instance:
            try:
                osc_server_instance.shutdown()
            except:
                pass

        # Close MIDI
        if midi_out:
            try:
                midi_out.close()
            except:
                pass

        print("Cleanup complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Create visualization window if MIDI is enabled
    if enable_midi:
        cv2.namedWindow("MIDI Control", cv2.WINDOW_NORMAL)

    try:
        frame_count = 0
        print("Main loop started...")

        while not shutdown_requested.is_set():
            frame_count += 1

            # MIDI processing
            if enable_midi:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Couldn't read webcam frame")
                    time.sleep(0.01)
                    continue

                # If in shared webcam mode, send frame to HackTV
                if use_shared_webcam:
                    try:
                        # Non-blocking put - drop frame if queue is full
                        frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        # Queue is full, skip this frame for HackTV
                        debug_print("Frame queue full, dropping frame for HackTV")

                # Update RGB channels with BGR frame (sends CC automatically)
                for channel in channels[:3]:  # Channels 1-3 (Red, Green, Blue)
                    if hasattr(channel, 'update_from_bgr'):
                        channel.update_from_bgr(frame)

                # Update total brightness channel with grayscale (sends CC automatically)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                channels[3].update(gray_frame)

                # Visualize
                vis_frame = frame.copy()

                y_offset = 30
                for i, channel in enumerate(channels):
                    channel_num = i + 1
                    color = CHANNEL_COLORS.get(i, (255, 255, 255))

                    if isinstance(channel, Channel1Red):
                        normalized = channel.current_brightness / BRIGHTNESS_RANGE
                        cc_val = int(CH1_MIN + normalized * (CH1_MAX - CH1_MIN))
                        status_text = f"T1 Red→Feedback: {channel.current_brightness:.1f} | CC{channel.cc_number}={cc_val} [{CH1_MIN}-{CH1_MAX}]"
                    elif isinstance(channel, Channel2Green):
                        normalized = channel.current_brightness / BRIGHTNESS_RANGE
                        cc_val = int(CH2_MIN + normalized * (CH2_MAX - CH2_MIN))
                        status_text = f"T2 Green→Drive: {channel.current_brightness:.1f} | CC{channel.cc_number}={cc_val} [{CH2_MIN}-{CH2_MAX}]"
                    elif isinstance(channel, Channel3Blue):
                        normalized = channel.current_brightness / BRIGHTNESS_RANGE
                        cc_val = int(CH3_MIN + normalized * (CH3_MAX - CH3_MIN))
                        status_text = f"T2 Blue→Filter: {channel.current_brightness:.1f} | CC{channel.cc_number}={cc_val} [{CH3_MIN}-{CH3_MAX}]"
                    elif isinstance(channel, Channel4Brightness):
                        normalized = channel.current_brightness / BRIGHTNESS_RANGE
                        cc_val = int(CH4_MIN + normalized * (CH4_MAX - CH4_MIN))
                        status_text = f"T4 Total→Volume: {channel.current_brightness:.1f} | CC{channel.cc_number}={cc_val} [{CH4_MIN}-{CH4_MAX}]"
                    else:
                        status_text = f"CH{channel_num}: (unknown)"

                    cv2.putText(
                        vis_frame, status_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
                    )
                    cv2.putText(
                        vis_frame, status_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                    )
                    y_offset += 30

                # Add queue status if in shared mode
                if use_shared_webcam:
                    queue_text = f"HackTV Queue: {frame_queue.qsize()}/5"
                    cv2.putText(
                        vis_frame, queue_text, (vis_frame.shape[1] - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA
                    )

                cv2.imshow("MIDI Control", vis_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("ESC pressed, exiting...")
                    break
            else:
                # If only HackTV is enabled, just wait
                time.sleep(0.1)

    except Exception as e:
        import traceback
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())

    finally:
        print("Cleaning up resources...")
        shutdown_requested.set()

        # Stop HackTV first
        stop_hacktv_pipeline()
        time.sleep(0.3)

        # Release camera
        if cap:
            try:
                cap.release()
            except:
                pass

        try:
            cv2.destroyAllWindows()
        except:
            pass

        # Close MIDI
        if midi_out:
            try:
                midi_out.close()
            except:
                pass

        print("Program finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MIDI from webcam and/or stream to HackTV"
    )
    parser.add_argument(
        "--midi-mapping",
        type=str,
        default="midi_mapping_digitone.csv",
        help="Path to MIDI mapping CSV file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--midi",
        action="store_true",
        default=True,
        help="Enable MIDI generation (default: enabled)"
    )
    parser.add_argument(
        "--no-midi",
        action="store_true",
        help="Disable MIDI generation"
    )
    parser.add_argument(
        "--hacktv",
        action="store_true",
        help="Enable HackTV streaming (default: disabled)"
    )
    parser.add_argument(
        "--webcam",
        type=str,
        default=None,
        help="Webcam device path (e.g., /dev/video0)"
    )
    parser.add_argument(
        "--midi-device",
        type=int,
        default=None,
        help="MIDI device index to use (see list at startup, default: 0)"
    )

    args = parser.parse_args()

    enable_midi = args.midi and not args.no_midi
    enable_hacktv = args.hacktv

    if not enable_midi and not enable_hacktv:
        print("Error: At least one of --midi or --hacktv must be enabled")
        sys.exit(1)

    print("Command line arguments:")
    print(f"  midi-mapping: {args.midi_mapping}")
    print(f"  debug: {args.debug}")
    print(f"  midi: {enable_midi}")
    print(f"  hacktv: {enable_hacktv}")
    print(f"  webcam: {args.webcam if args.webcam else 'auto-detect'}")
    print(f"  midi-device: {args.midi_device if args.midi_device is not None else 'auto (first device)'}")

    main(
        args.midi_mapping,
        args.debug,
        enable_midi,
        enable_hacktv,
        args.webcam,
        args.midi_device
    )
