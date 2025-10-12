#!/usr/bin/env python3
"""
Stream webcam mixed with video loop to HackRF TV transmitter
with TRUE real-time mixing control that restarts pipeline on change
"""

import subprocess
import sys
import signal
import time
import threading
import os
from pathlib import Path

try:
    from pythonosc import dispatcher, osc_server
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False

# Configuration
WEBCAM_DEVICE = None  # Will be auto-detected
VIDEO_LOOP_FILE = "Teaser05.mp4"
VIDEO_RESOLUTION = "720x576"  # PAL resolution
VIDEO_FRAMERATE = 25

# HackTV settings
HACKTV_MODE = "g"
HACKTV_FREQ = "471250000"
HACKTV_SAMPLE_RATE = "16000000"
HACKTV_GAIN = "47"

# OSC settings
OSC_IP = "0.0.0.0"
OSC_PORT = 8000

# Process handles
processes = []
osc_server_instance = None
restart_requested = threading.Event()
shutdown_requested = threading.Event()

# Mixing state
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

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nStopping streams...")
    shutdown_requested.set()

    if osc_server_instance:
        try:
            osc_server_instance.shutdown()
        except:
            pass

    for proc in processes:
        try:
            proc.terminate()
        except:
            pass
    time.sleep(1)
    for proc in processes:
        try:
            if proc.poll() is None:
                proc.kill()
        except:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

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

def osc_mix_handler(address, *args):
    """Handle OSC messages for mix control"""
    if len(args) > 0:
        mix_value = float(args[0])
        new_mix, old_mix = mixer_state.set_mix(mix_value)
        if abs(new_mix - old_mix) > 0.01:
            restart_requested.set()
        print(f"\rMix: {new_mix:.2f} (Webcam ← → Loop) - Restarting...  ", end="", flush=True)

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
        print(f"OSC server listening on {OSC_IP}:{OSC_PORT}")
        print(f"Send OSC to /mix with value 0.0-1.0")

        server_thread = threading.Thread(target=osc_server_instance.serve_forever)
        server_thread.daemon = True
        server_thread.start()
    except Exception as e:
        print(f"Could not start OSC server: {e}")

def keyboard_control_thread():
    """Monitor keyboard input for mix control with debouncing"""
    print("\nKeyboard controls:")
    print("  ← → : Adjust mix (±5%)")
    print("  1-9 : Set mix presets")
    print("  0   : 50/50 mix")
    print("  Q   : All webcam (0%)")
    print("  W   : All loop (100%)")
    print("  R   : Force restart pipeline")
    print("\nNote: Changes apply 1 second after you stop adjusting")

    last_change_time = [0]  # Use list for mutable closure

    # Debounce watcher thread
    def debounce_watcher():
        while not shutdown_requested.is_set():
            if last_change_time[0] > 0 and (time.time() - last_change_time[0]) >= 1.0:
                if not restart_requested.is_set():
                    print(f"\rMix: {mixer_state.get_mix():.2f} - Applying...     ", end="", flush=True)
                    restart_requested.set()
                last_change_time[0] = 0
            time.sleep(0.1)

    debounce_thread = threading.Thread(target=debounce_watcher)
    debounce_thread.daemon = True
    debounce_thread.start()

    try:
        import tty
        import termios

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())

            while not shutdown_requested.is_set():
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
                    restart_requested.set()
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
        print("(Advanced keyboard control not available)")
        pass

def build_ffmpeg_command():
    """Build FFmpeg command with current mix value"""
    mix = mixer_state.get_mix()
    width, height = VIDEO_RESOLUTION.split('x')

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-f", "v4l2",
        "-input_format", "mjpeg",
        "-framerate", "25",
        "-video_size", "640x480",
        "-i", WEBCAM_DEVICE,
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

    return cmd

def start_pipeline():
    """Start FFmpeg and HackTV pipeline"""
    global processes

    # Build commands
    ffmpeg_cmd = build_ffmpeg_command()
    hacktv_cmd = [
        "hacktv",
        "-m", HACKTV_MODE,
        "-f", HACKTV_FREQ,
        "-s", HACKTV_SAMPLE_RATE,
        "-g", HACKTV_GAIN,
        "--repeat",
        "-"
    ]

    # Start pipeline
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    hacktv_proc = subprocess.Popen(
        hacktv_cmd,
        stdin=ffmpeg_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    ffmpeg_proc.stdout.close()

    processes = [ffmpeg_proc, hacktv_proc]
    return processes

def stop_pipeline():
    """Stop current pipeline quickly"""
    global processes

    for proc in processes:
        try:
            proc.terminate()
        except:
            pass

    time.sleep(0.1)  # Reduced delay

    for proc in processes:
        try:
            if proc.poll() is None:
                proc.kill()
        except:
            pass

    processes = []

def main():
    global WEBCAM_DEVICE

    if not Path(VIDEO_LOOP_FILE).exists():
        print(f"Error: Video loop file '{VIDEO_LOOP_FILE}' not found!")
        sys.exit(1)

    print("=" * 60)
    print("WEBCAM + VIDEO LOOP MIXER → HACKTV (DYNAMIC)")
    print("=" * 60)

    if WEBCAM_DEVICE is None:
        print("\nDetecting webcam...")
        WEBCAM_DEVICE = find_webcam()
        if WEBCAM_DEVICE is None:
            print("\nError: No webcam device found!")
            sys.exit(1)

    print(f"\n✓ Webcam: {WEBCAM_DEVICE}")
    print(f"✓ Loop video: {VIDEO_LOOP_FILE}")
    print(f"✓ TV frequency: {HACKTV_FREQ} Hz (PAL {HACKTV_MODE.upper()})")
    print(f"✓ Initial mix: {mixer_state.get_mix():.2f}")
    print("\nPress Ctrl+C to stop\n")

    if OSC_AVAILABLE:
        start_osc_server()

    control_thread = threading.Thread(target=keyboard_control_thread)
    control_thread.daemon = True
    control_thread.start()

    try:
        # Start initial pipeline
        print(f"Starting with mix: {mixer_state.get_mix():.2f}\n")
        start_pipeline()
        print("✓ Streaming active!\n")

        # Main loop - restart on mix change
        while not shutdown_requested.is_set():
            # Check if restart requested
            if restart_requested.is_set():
                restart_requested.clear()
                print(f"\n🔄 Restarting with new mix: {mixer_state.get_mix():.2f}...")
                stop_pipeline()
                time.sleep(0.05)  # Minimal delay
                start_pipeline()
                print("✓ Streaming resumed\n")

            # Check if processes died
            for i, proc in enumerate(processes):
                if proc.poll() is not None:
                    proc_name = "FFmpeg" if i == 0 else "HackTV"
                    print(f"\n⚠ {proc_name} process died! Restarting...")
                    stop_pipeline()
                    time.sleep(1)
                    start_pipeline()
                    break

            time.sleep(0.2)

    except Exception as e:
        print(f"Error: {e}")
        signal_handler(None, None)

if __name__ == "__main__":
    main()
