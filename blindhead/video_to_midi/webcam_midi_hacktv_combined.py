#!/usr/bin/env python3
"""
Combined script: Generate MIDI from webcam AND stream to HackTV
- Uses single webcam feed for both MIDI generation and TV transmission
- MIDI generation runs independently in main thread
- HackTV video mixing runs in subprocess pipeline
- Both can be enabled/disabled via command line flags
"""

import argparse
import platform
import signal
import time
import subprocess
import threading
import sys
from pathlib import Path
from collections import deque

import cv2
import mido
import numpy as np
import pandas as pd

try:
    from pythonosc import dispatcher, osc_server
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# Constants
MIDI_RANGE = 127
BRIGHTNESS_RANGE = 255
NUM_CHANNELS = 4
DEBUG = False

# MIDI limiting and visualization defaults
MIDI_LIMITING_ENABLED = True
MIDI_GLOBAL_MAX_RATE = 200            # messages per second (token bucket fill rate)
MIDI_PER_CHANNEL_MIN_INTERVAL_S = 0.01  # 10 ms between sends per (channel,cc)
MIDI_QUANTIZE_STEP = 2                # send only on >= step change in CC value
MIDI_GRAPH_HISTORY_SECONDS = 20       # seconds of history for overlay graph


# Channel configurations
# Each tuple: (name, color_channel, cc_number, midi_channel, min_value, max_value)
# color_channel: 'red', 'green', 'blue', or 'gray' for total brightness
CHANNEL_CONFIGS = [
    ("Red→T1 Feedback", "red", 19, 0, 0, 127),      # Track 1 SYN 1 Feedback
    ("Green→T4 LFO Speed", "green", 23, 3, 0, 127),  # Track 4 Filter Frequency 
    ("Blue→T2 Filter", "blue", 23, 1, 50, 100),      # Track 2 Filter Frequency
    ("Total→T3 SYN1 Mix", "gray", 23, 2, 0, 127),    # Track 3 Filter Frequency
]

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
frame_queue = queue.Queue(maxsize=30)  # Larger buffer for smoother HackTV streaming
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


class ColorChannel:
    """Generic channel that extracts color/brightness and sends MIDI CC"""

    def __init__(self, name, color_channel, cc_number, midi_channel, min_value, max_value, midi_out,
                 limiter=None, stats=None, quantize_step=1):
        """
        Args:
            name: Display name for this channel
            color_channel: 'red', 'green', 'blue', or 'gray' for total brightness
            cc_number: MIDI CC number to send
            midi_channel: MIDI channel (0-indexed)
            min_value: Minimum CC value
            max_value: Maximum CC value
            midi_out: MIDI output object
        """
        self.name = name
        self.color_channel = color_channel
        self.cc_number = cc_number
        self.midi_channel = midi_channel
        self.min_value = min_value
        self.max_value = max_value
        self.midi_out = midi_out
        self.current_brightness = 0
        self.smoothing_factor = 0.3
        self.last_sent_value = -1
        self.quantize_step = max(1, int(quantize_step))
        self.limiter = limiter
        self.stats = stats

    def update(self, frame):
        """Update from frame (handles both BGR and grayscale)"""
        # Extract the appropriate channel
        if self.color_channel == 'gray':
            # Expect grayscale frame
            avg_brightness = np.mean(frame)
        else:
            # Expect BGR frame, extract specific color
            color_index = {'blue': 0, 'green': 1, 'red': 2}[self.color_channel]
            avg_brightness = np.mean(frame[:, :, color_index])

        # Apply smoothing
        self.current_brightness = (
            self.current_brightness * (1 - self.smoothing_factor) +
            avg_brightness * self.smoothing_factor
        )
        debug_print(f"{self.name} brightness: {self.current_brightness:.2f}")

        # Send CC message
        self._send_cc()

    def _send_cc(self):
        """Send CC message based on current brightness"""
        if not self.midi_out:
            return

        # Map brightness (0-255) to configured min-max range
        normalized = self.current_brightness / BRIGHTNESS_RANGE
        raw_value = int(self.min_value + normalized * (self.max_value - self.min_value))
        raw_value = max(self.min_value, min(self.max_value, raw_value))

        # Quantize value
        if self.quantize_step > 1:
            # snap to nearest multiple within [min, max]
            q = int(round(raw_value / self.quantize_step) * self.quantize_step)
            cc_value = max(self.min_value, min(self.max_value, q))
        else:
            cc_value = raw_value

        # Only attempt if value changed
        if cc_value != self.last_sent_value:
            if self.stats is not None:
                self.stats.record_attempt()

            allowed = True
            if self.limiter is not None:
                allowed = self.limiter.allow((self.midi_channel, self.cc_number))

            if allowed:
                self.midi_out.send(
                    mido.Message(
                        'control_change',
                        control=self.cc_number,
                        value=cc_value,
                        channel=self.midi_channel
                    )
                )
                if self.stats is not None:
                    self.stats.record_sent()
                self.last_sent_value = cc_value
                debug_print(f"{self.name} CC{self.cc_number}: {cc_value}")


# Rate limiting and stats helpers
class MidiLimiter:
    def __init__(self, enabled=True, global_rate=MIDI_GLOBAL_MAX_RATE, per_chan_interval=MIDI_PER_CHANNEL_MIN_INTERVAL_S):
        self.enabled = enabled
        self.global_rate = float(global_rate)
        self.capacity = max(1.0, self.global_rate * 1.5)
        self.tokens = self.capacity
        self.last_refill = time.time()
        self.per_chan_interval = float(per_chan_interval)
        self.last_sent_time = {}

    def allow(self, chan_key):
        if not self.enabled:
            return True
        now = time.time()
        # Refill tokens
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.global_rate)
            self.last_refill = now
        # Per-channel interval check
        last_t = self.last_sent_time.get(chan_key, 0.0)
        if (now - last_t) < self.per_chan_interval:
            return False
        # Token check
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            self.last_sent_time[chan_key] = now
            return True
        return False


class MidiStats:
    def __init__(self, history_seconds=MIDI_GRAPH_HISTORY_SECONDS):
        self.attempted_cur = 0
        self.sent_cur = 0
        self.last_tick = time.time()
        self.history_attempted = deque(maxlen=int(history_seconds))
        self.history_sent = deque(maxlen=int(history_seconds))

    def record_attempt(self):
        self.attempted_cur += 1

    def record_sent(self):
        self.sent_cur += 1

    def tick_second(self):
        now = time.time()
        if (now - self.last_tick) >= 1.0:
            self.history_attempted.append(self.attempted_cur)
            self.history_sent.append(self.sent_cur)
            self.attempted_cur = 0
            self.sent_cur = 0
            self.last_tick = now


def _load_arial_font(size=14):
    if not PIL_AVAILABLE:
        return None
    candidates = [
        "/Library/Fonts/Arial.ttf",                # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS newer
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",  # Linux
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",   # fallback
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_midi_rate_graph(frame, stats: MidiStats, limiting_enabled: bool, max_rate: int):
    try:
        hist_attempt = list(stats.history_attempted)
        hist_sent = list(stats.history_sent)
        if not hist_attempt and not hist_sent:
            return frame

        # Graph area
        h, w = frame.shape[:2]
        graph_w = 260
        graph_h = 120
        margin = 12
        x0 = w - graph_w - margin
        y0 = h - graph_h - margin
        x1 = w - margin
        y1 = h - margin

        # Prepare background box
        cv2.rectangle(frame, (x0, y0), (x1, y1), (20, 20, 20), thickness=-1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 80, 80), thickness=1)

        # Determine scaling (use max of observed and limit)
        max_observed = max(hist_attempt + hist_sent) if (hist_attempt or hist_sent) else 1
        max_val = max([1, max_observed] + ([max_rate] if limiting_enabled and max_rate > 0 else []))

        # Reserve space at top for legend/title so lines/labels never cross it
        legend_reserved_px = 48  # keep lines and y-ticks below this
        top_plot_y = y0 + legend_reserved_px
        bottom_plot_y = y1 - 10
        usable_h = max(1, bottom_plot_y - top_plot_y)
        scale_y = usable_h / float(max_val)

        # Helper to plot series
        def plot_series(values, color, mode='line', thickness=2):
            if len(values) < 1:
                return
            step_x = max(1, int((graph_w - 20) / max(1, len(values) - 1)))
            pts = []
            for i, v in enumerate(values[-int((graph_w - 20)/step_x) - 1:]):
                x = x0 + 10 + i * step_x
                y = bottom_plot_y - int(v * scale_y)
                pts.append((x, y))
            if mode == 'line' and len(pts) >= 2:
                cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
            elif mode == 'markers':
                for p in pts:
                    cv2.circle(frame, p, max(1, thickness // 2), color, -1, lineType=cv2.LINE_AA)

        # Draw sent (green line, thicker) and attempts (yellow markers) for clarity
        plot_series(hist_sent, (0, 200, 0), mode='line', thickness=3)
        plot_series(hist_attempt, (0, 255, 255), mode='markers', thickness=2)

        # Draw rate limit line if enabled (keep well below legend area)
        if limiting_enabled and max_rate > 0:
            y_lim = bottom_plot_y - int(max_rate * scale_y)
            cv2.line(frame, (x0 + 10, y_lim), (x1 - 10, y_lim), (0, 0, 255), 1, cv2.LINE_AA)

        # Labels using PIL for Arial if available
        attempted_last = hist_attempt[-1] if hist_attempt else 0
        sent_last = hist_sent[-1] if hist_sent else 0
        title = f"MIDI messages per second  (attempt {attempted_last}  sent {sent_last})"
        legend1 = "attempt"
        legend2 = "sent"

        if PIL_AVAILABLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            font = _load_arial_font(14)
            draw.text((x0 + 10, y0 + 6), title, fill=(220, 220, 220), font=font)
            draw.rectangle([x0 + 10, y0 + 28, x0 + 30, y0 + 33], fill=(255, 255, 0))
            draw.text((x0 + 34, y0 + 24), legend1, fill=(230, 230, 230), font=font)
            draw.rectangle([x0 + 100, y0 + 28, x0 + 120, y0 + 33], fill=(0, 220, 0))
            draw.text((x0 + 124, y0 + 24), legend2, fill=(230, 230, 230), font=font)
            # Y-axis ticks and labels (0, limit if enabled, max observed)
            ticks = [(0, "0")]
            if limiting_enabled and max_rate > 0:
                ticks.append((max_rate, f"{max_rate}"))
            ticks.append((max_observed, f"{max_observed}"))
            for val, lbl in ticks:
                y_tick = bottom_plot_y - int(val * scale_y)
                draw.line([(x0 + 8, y_tick), (x0 + 10, y_tick)], fill=(200, 200, 200), width=1)
                draw.text((x0 + 2, y_tick - 8), lbl, fill=(200, 200, 200), font=font)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            cv2.putText(frame, title, (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x0 + 10, y0 + 28), (x0 + 30, y0 + 33), (0, 255, 255), thickness=-1)
            cv2.putText(frame, legend1, (x0 + 34, y0 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x0 + 100, y0 + 28), (x0 + 120, y0 + 33), (0, 200, 0), thickness=-1)
            cv2.putText(frame, legend2, (x0 + 124, y0 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
            # OpenCV-only y-axis rough ticks
            cv2.putText(frame, "0", (x0 + 2, bottom_plot_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            if limiting_enabled and max_rate > 0:
                y_tick = bottom_plot_y - int(max_rate * scale_y)
                cv2.line(frame, (x0 + 8, y_tick), (x0 + 10, y_tick), (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{max_rate}", (x0 + 2, y_tick - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{max_observed}", (x0 + 2, top_plot_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        # X-axis (time) at bottom of graph
        cv2.line(frame, (x0 + 10, bottom_plot_y), (x1 - 10, bottom_plot_y), (120, 120, 120), 1, cv2.LINE_AA)

        # X-axis labels: left shows "-Ns", right shows "now"
        n_seconds = max(len(hist_attempt), len(hist_sent))
        if n_seconds > 0:
            cv2.putText(frame, "now", (x1 - 45, bottom_plot_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"-{n_seconds}s", (x0 + 10, bottom_plot_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    except Exception:
        # Fail-safe: never crash visualization
        pass
    return frame
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
    system = platform.system()
    
    # macOS/Darwin detection
    if system == "Darwin":
        print("Detected macOS - testing webcam with OpenCV...")
        
        # Try common indices (0 is usually built-in camera)
        for index in range(5):
            print(f"\nTesting webcam index {index}...", end=" ")
            cap = cv2.VideoCapture(index)
            
            if cap.isOpened():
                # Try to read a frame to verify it works
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    print(f"✓ Working camera found")
                    response = input(f"Use webcam index {index}? (y/n): ").lower()
                    if response == 'y':
                        return index
                else:
                    print("✗ Could not read frame")
            else:
                print("✗ Not available")
        
        # Manual input
        manual_index = input("\nEnter webcam index manually (usually 0): ").strip()
        if manual_index.isdigit():
            return int(manual_index)
        
        return None
    
    # Linux detection
    elif system == "Linux":
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
    
    else:
        print(f"Unsupported platform: {system}")
        print("Trying default webcam index 0...")
        return 0


def build_ffmpeg_command(webcam_device, use_stdin=False, frame_width=640, frame_height=480):
    """Build FFmpeg command with current mix value

    Args:
        webcam_device: Path to webcam device or index (only used if use_stdin=False)
        use_stdin: If True, accept rawvideo from stdin instead of v4l2/avfoundation
        frame_width: Width of input frames (when use_stdin=True)
        frame_height: Height of input frames (when use_stdin=True)
    """
    mix = mixer_state.get_mix()
    width, height = VIDEO_RESOLUTION.split('x')
    system = platform.system()

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
    elif system == "Darwin":
        # macOS: Use AVFoundation
        cmd = [
            "ffmpeg",
            "-loglevel", "info",
            "-f", "avfoundation",
            "-framerate", "25",
            "-video_size", "640x480",
            "-i", str(webcam_device),
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
        # Linux: Use v4l2 capture mode
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
        print(f"Platform: {system}")
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
    timing_diagnostics=False,
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
    midi_stats = None
    midi_limiter = None

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

        # Create limiter and stats
        midi_stats = MidiStats(history_seconds=MIDI_GRAPH_HISTORY_SECONDS)
        midi_limiter = MidiLimiter(
            enabled=MIDI_LIMITING_ENABLED,
            global_rate=MIDI_GLOBAL_MAX_RATE,
            per_chan_interval=MIDI_PER_CHANNEL_MIN_INTERVAL_S,
        )

        # Create channels from configuration
        channels = []
        for name, color_ch, cc_num, midi_ch, min_val, max_val in CHANNEL_CONFIGS:
            channel = ColorChannel(
                name, color_ch, cc_num, midi_ch, min_val, max_val, midi_out,
                limiter=midi_limiter, stats=midi_stats, quantize_step=MIDI_QUANTIZE_STEP
            )
            channels.append(channel)

        print("MIDI channels initialized")
        for i, (name, color_ch, cc_num, midi_ch, min_val, max_val) in enumerate(CHANNEL_CONFIGS):
            print(f"Channel {i+1}: {name} (CC {cc_num}) → MIDI Ch {midi_ch+1} [{min_val}-{max_val}]")

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
        last_frame_time = time.time()
        target_fps = 30  # Limit to 30 FPS to reduce CPU load
        frame_interval = 1.0 / target_fps

        # Timing diagnostics (only if enabled)
        timing_samples = []
        timing_report_interval = 100  # Report every 100 frames

        print("Main loop started...")

        while not shutdown_requested.is_set():
            frame_count += 1

            # MIDI processing
            if enable_midi:
                # Timing: Frame capture
                if timing_diagnostics:
                    t0 = time.time()

                ret, frame = cap.read()
                if not ret:
                    print("Warning: Couldn't read webcam frame")
                    time.sleep(0.01)
                    continue

                if timing_diagnostics:
                    t1 = time.time()

                # If in shared webcam mode, send frame to HackTV
                if use_shared_webcam:
                    try:
                        # Non-blocking put - reuse frame without copying
                        frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Queue is full, skip this frame for HackTV
                        debug_print("Frame queue full, dropping frame for HackTV")

                if timing_diagnostics:
                    t2 = time.time()

                # Update all channels
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if timing_diagnostics:
                    t3 = time.time()

                for channel in channels:
                    if channel.color_channel == 'gray':
                        channel.update(gray_frame)
                    else:
                        channel.update(frame)

                if timing_diagnostics:
                    t4 = time.time()

                # Visualize only every other frame to reduce overhead
                if frame_count % 2 == 0:
                    vis_frame = frame.copy()

                    y_offset = 30
                    for i, channel in enumerate(channels):
                        color = CHANNEL_COLORS.get(i, (255, 255, 255))

                        # Calculate current CC value
                        normalized = channel.current_brightness / BRIGHTNESS_RANGE
                        cc_val = int(channel.min_value + normalized * (channel.max_value - channel.min_value))
                        cc_val = max(channel.min_value, min(channel.max_value, cc_val))

                        status_text = f"{channel.name}: {channel.current_brightness:.1f} | CC{channel.cc_number}={cc_val} [{channel.min_value}-{channel.max_value}]"

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
                        queue_text = f"HackTV Queue: {frame_queue.qsize()}/30"
                        cv2.putText(
                            vis_frame, queue_text, (vis_frame.shape[1] - 300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA
                        )

                    # Draw MIDI throughput graph
                    if midi_stats is not None:
                        vis_frame = draw_midi_rate_graph(
                            vis_frame,
                            midi_stats,
                            limiting_enabled=MIDI_LIMITING_ENABLED,
                            max_rate=int(MIDI_GLOBAL_MAX_RATE),
                        )

                    cv2.imshow("MIDI Control", vis_frame)

                if timing_diagnostics:
                    t5 = time.time()

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("ESC pressed, exiting...")
                    break

                if timing_diagnostics:
                    t6 = time.time()

                    # Collect timing data
                    timing_data = {
                        'capture': (t1 - t0) * 1000,
                        'queue': (t2 - t1) * 1000,
                        'gray_convert': (t3 - t2) * 1000,
                        'midi_update': (t4 - t3) * 1000,
                        'visualization': (t5 - t4) * 1000 if frame_count % 2 == 0 else 0,
                        'waitkey': (t6 - t5) * 1000,
                        'total_processing': (t6 - t0) * 1000
                    }
                    timing_samples.append(timing_data)

                    # Report timing every N frames
                    if frame_count % timing_report_interval == 0 and timing_samples:
                        avg_timings = {
                            key: sum(t[key] for t in timing_samples) / len(timing_samples)
                            for key in timing_samples[0].keys()
                        }
                        print(f"\n=== Timing Report (frame {frame_count}) ===")
                        print(f"  Frame capture:    {avg_timings['capture']:.2f} ms")
                        print(f"  Queue frame:      {avg_timings['queue']:.2f} ms")
                        print(f"  Gray convert:     {avg_timings['gray_convert']:.2f} ms")
                        print(f"  MIDI update:      {avg_timings['midi_update']:.2f} ms")
                        print(f"  Visualization:    {avg_timings['visualization']:.2f} ms")
                        print(f"  cv2.waitKey:      {avg_timings['waitkey']:.2f} ms")
                        print(f"  Total processing: {avg_timings['total_processing']:.2f} ms")
                        print(f"  Effective FPS:    {1000/avg_timings['total_processing']:.1f}")
                        timing_samples = []  # Reset for next batch

                # Frame rate limiting
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_frame_time = time.time()

                # Stats tick
                if midi_stats is not None:
                    midi_stats.tick_second()
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
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Enable timing diagnostics (reports every 100 frames)"
    )
    parser.add_argument(
        "--limit-midi",
        action="store_true",
        default=True,
        help="Enable MIDI rate limiting (default: enabled)"
    )
    parser.add_argument(
        "--no-limit-midi",
        action="store_true",
        help="Disable MIDI rate limiting"
    )
    parser.add_argument(
        "--midi-max-rate",
        type=int,
        default=MIDI_GLOBAL_MAX_RATE,
        help="Global max MIDI CC messages per second"
    )
    parser.add_argument(
        "--midi-per-chan-interval-ms",
        type=int,
        default=int(MIDI_PER_CHANNEL_MIN_INTERVAL_S * 1000),
        help="Minimum interval between CCs per (channel,cc) in milliseconds"
    )
    parser.add_argument(
        "--midi-quantize-step",
        type=int,
        default=MIDI_QUANTIZE_STEP,
        help="Quantization step for CC values (1 disables quantization)"
    )
    parser.add_argument(
        "--graph-history-seconds",
        type=int,
        default=MIDI_GRAPH_HISTORY_SECONDS,
        help="Seconds of throughput history to show in the overlay"
    )

    args = parser.parse_args()

    enable_midi = args.midi and not args.no_midi
    enable_hacktv = args.hacktv

    # Apply MIDI limiting configuration
    MIDI_LIMITING_ENABLED = bool(args.limit_midi and not args.no_limit_midi)
    MIDI_GLOBAL_MAX_RATE = max(1, int(args.midi_max_rate))
    MIDI_PER_CHANNEL_MIN_INTERVAL_S = max(0.0, float(args.midi_per_chan_interval_ms) / 1000.0)
    MIDI_QUANTIZE_STEP = max(1, int(args.midi_quantize_step))
    MIDI_GRAPH_HISTORY_SECONDS = max(1, int(args.graph_history_seconds))

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
    print(f"  timing: {args.timing}")
    print(f"  midi limiting: {MIDI_LIMITING_ENABLED} | max_rate={MIDI_GLOBAL_MAX_RATE} | per_chan_interval_ms={int(MIDI_PER_CHANNEL_MIN_INTERVAL_S*1000)} | quant_step={MIDI_QUANTIZE_STEP}")
    print(f"  graph history: {MIDI_GRAPH_HISTORY_SECONDS}s")

    main(
        args.midi_mapping,
        args.debug,
        enable_midi,
        enable_hacktv,
        args.webcam,
        args.midi_device,
        args.timing
    )
