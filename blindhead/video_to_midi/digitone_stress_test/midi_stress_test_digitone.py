#!/usr/bin/env python3
"""
Digitone MIDI Stress Test

Purpose:
- Stress test Elektron Digitone's MIDI input via Scarlett 6i6 DIN (through a computer's MIDI out)
- Explore axes that could have caused freezes: CC rate, burstiness, number of streams, quantization, patterns

Features:
- Profiles:
  - current-four: CCs and channels from webcam_midi_hacktv_combined.py
  - sweep-csv: load CCs (and optional channels) from midi_mapping_digitone.csv
- Precise scheduler with steady or burst modes
- Patterns: ramp, triangle, random, step
- Optional quantization and duplicate suppression
- CSV logging and periodic stats

Python 3.10.12 compatible, with full type hints for functions.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import mido

try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False


# =============================
# Profiles and defaults
# =============================

CURRENT_FOUR_PROFILE: List[Tuple[int, int]] = [
    # (cc, midi_channel)
    (19, 0),  # Track 1 Feedback
    (28, 3),  # Track 4 LFO Speed
    (23, 1),  # Track 2 Filter
    (21, 2),  # Track 3 SYN1 Mix
]


# =============================
# Utilities
# =============================

def list_midi_outputs() -> List[str]:
    return mido.get_output_names()


def open_midi_output_by_index_or_name(
    index: Optional[int], name: Optional[str]
) -> Optional[mido.ports.BaseOutput]:
    outputs = list_midi_outputs()
    if not outputs:
        print("No MIDI outputs available.")
        return None

    selected: Optional[str] = None
    if name:
        for out in outputs:
            if name.lower() in out.lower():
                selected = out
                break
        if selected is None:
            print(f"MIDI output matching name '{name}' not found. Available:")
            for i, n in enumerate(outputs):
                print(f"  [{i}] {n}")
            return None
    elif index is not None:
        if 0 <= index < len(outputs):
            selected = outputs[index]
        else:
            print(f"MIDI output index {index} out of range 0..{len(outputs)-1}.")
            return None
    else:
        selected = outputs[0]

    try:
        port = mido.open_output(selected)
        print(f"Using MIDI output: {selected}")
        return port
    except Exception as exc:  # noqa: BLE001 - surface errors explicitly
        print(f"Error opening MIDI output '{selected}': {exc}")
        return None


def parse_int_list(spec: Optional[str]) -> Optional[List[int]]:
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    result: List[int] = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            raise ValueError(f"Invalid integer in list: '{p}'")
    return result


def clamp(value: int, minimum: int, maximum: int) -> int:
    return minimum if value < minimum else maximum if value > maximum else value


def quantize_value(value: int, step: int, minimum: int, maximum: int) -> int:
    if step <= 1:
        return clamp(value, minimum, maximum)
    value = clamp(value, minimum, maximum)
    # Snap to nearest multiple of step within [minimum, maximum]
    snapped = int(round(value / step) * step)
    return clamp(snapped, minimum, maximum)


# =============================
# Mapping loading
# =============================

def load_profile_current_four() -> List[Tuple[int, int]]:
    return CURRENT_FOUR_PROFILE.copy()


def load_profile_from_csv(
    csv_path: Path,
    default_channel: Optional[int],
    limit: Optional[int],
) -> List[Tuple[int, int]]:
    """Load (cc, channel) pairs from a CSV.

    Accepts flexible column names (case-insensitive): cc, control, cc_number, channel, midi_channel
    If channel is missing, use default_channel or 0.
    """
    pairs: List[Tuple[int, int]] = []
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    def normalize_col(name: str) -> str:
        return name.strip().lower().replace(' ', '_')

    if PANDAS_AVAILABLE:
        df = pd.read_csv(csv_path)  # type: ignore
        cols = {normalize_col(c): c for c in df.columns}
        cc_col = None
        for candidate in ("cc", "control", "cc_number"):
            if candidate in cols:
                cc_col = cols[candidate]
                break
        if cc_col is None:
            raise ValueError("CSV missing CC column (expected one of: cc, control, cc_number)")
        ch_col = None
        for candidate in ("channel", "midi_channel"):
            if candidate in cols:
                ch_col = cols[candidate]
                break

        for _, row in df.iterrows():
            try:
                cc_val = int(row[cc_col])
            except Exception:
                continue
            if not (0 <= cc_val <= 127):
                continue
            if ch_col is not None:
                try:
                    ch_val = int(row[ch_col])
                except Exception:
                    ch_val = default_channel if default_channel is not None else 0
            else:
                ch_val = default_channel if default_channel is not None else 0
            ch_val = clamp(ch_val, 0, 15)
            pairs.append((cc_val, ch_val))
            if limit is not None and len(pairs) >= limit:
                break
    else:
        with csv_path.open('r', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row")
            fields = [normalize_col(c) for c in reader.fieldnames]
            index_map = {normalize_col(c): i for i, c in enumerate(reader.fieldnames)}
            def get_column(*names: str) -> Optional[str]:
                for n in names:
                    if n in fields:
                        # Return original name
                        return reader.fieldnames[index_map[n]]
                return None

            cc_name = get_column("cc", "control", "cc_number")
            if cc_name is None:
                raise ValueError("CSV missing CC column (expected one of: cc, control, cc_number)")
            ch_name = get_column("channel", "midi_channel")

            for row in reader:
                try:
                    cc_val = int(row[cc_name])
                except Exception:
                    continue
                if not (0 <= cc_val <= 127):
                    continue
                if ch_name is not None:
                    try:
                        ch_val = int(row[ch_name])
                    except Exception:
                        ch_val = default_channel if default_channel is not None else 0
                else:
                    ch_val = default_channel if default_channel is not None else 0
                ch_val = clamp(ch_val, 0, 15)
                pairs.append((cc_val, ch_val))
                if limit is not None and len(pairs) >= limit:
                    break

    if not pairs:
        raise ValueError("No (cc, channel) pairs parsed from CSV")
    return pairs


# =============================
# Patterns
# =============================

PatternFunc = Callable[[int, int, int, int], int]


def pattern_ramp(prev: int, minimum: int, maximum: int, step: int) -> int:
    nxt = prev + step
    if nxt > maximum:
        span = maximum - minimum + 1
        if span <= 0:
            return minimum
        # wrap-around
        offs = (nxt - minimum) % span
        return minimum + offs
    return nxt


def pattern_triangle(prev: int, minimum: int, maximum: int, step: int, direction: int) -> Tuple[int, int]:
    nxt = prev + (step if direction > 0 else -step)
    if nxt >= maximum:
        return maximum, -1
    if nxt <= minimum:
        return minimum, 1
    return nxt, direction


def pattern_random(_: int, minimum: int, maximum: int, __: int) -> int:
    return random.randint(minimum, maximum)


def pattern_step(prev: int, minimum: int, maximum: int, step: int) -> int:
    # Toggle between bounds; step is ignored
    return maximum if prev <= minimum else minimum


# =============================
# Scheduler
# =============================

@dataclass
class StreamState:
    cc: int
    channel: int
    current_value: int
    next_time: float
    direction: int  # +1 or -1 (used for triangle)


def send_cc(
    port: mido.ports.BaseOutput,
    cc: int,
    channel: int,
    value: int,
) -> None:
    port.send(mido.Message('control_change', control=cc, value=value, channel=channel))


def run_stress_test(
    port: mido.ports.BaseOutput,
    pairs: List[Tuple[int, int]],
    duration_s: float,
    rate_total: Optional[float],
    rate_per_cc: Optional[float],
    pattern_name: str,
    quant_step: int,
    suppress_duplicates: bool,
    value_min: int,
    value_max: int,
    burst_on_ms: int,
    burst_off_ms: int,
    csv_log_path: Optional[Path],
    seed: Optional[int],
) -> None:
    if seed is not None:
        random.seed(seed)

    if rate_per_cc is None and rate_total is None:
        rate_total = 200.0

    num_streams = len(pairs)
    per_cc_rate: float
    if rate_per_cc is not None:
        per_cc_rate = float(rate_per_cc)
    else:
        per_cc_rate = float(rate_total) / max(1, num_streams)  # type: ignore[arg-type]

    interval_per_cc = 1.0 / max(0.001, per_cc_rate)  # avoid div by zero

    # Pattern function selection
    pattern_lower = pattern_name.lower()
    pattern_func: Optional[PatternFunc] = None
    triangle_mode = False
    if pattern_lower == 'ramp':
        pattern_func = pattern_ramp  # type: ignore[assignment]
    elif pattern_lower == 'triangle':
        triangle_mode = True
    elif pattern_lower == 'random':
        pattern_func = pattern_random  # type: ignore[assignment]
    elif pattern_lower == 'step':
        pattern_func = pattern_step  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")

    now = time.perf_counter()
    states: List[StreamState] = []
    for i, (cc, ch) in enumerate(pairs):
        start_val = value_min if pattern_lower in ("ramp", "triangle", "step") else random.randint(value_min, value_max)
        states.append(StreamState(cc=cc, channel=ch, current_value=start_val, next_time=now + i * 0.002, direction=+1))

    # Logging setup
    csv_file = None
    csv_writer = None
    if csv_log_path is not None:
        csv_file = csv_log_path.open('w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["t_rel_s", "iso", "cc", "channel", "value"])  # header

    # Stats
    last_stats_time = now
    sent_counts_total = 0
    sent_counts_per_stream: List[int] = [0 for _ in states]

    # Burst control
    burst_on_s = burst_on_ms / 1000.0 if burst_on_ms > 0 else 0.0
    burst_off_s = burst_off_ms / 1000.0 if burst_off_ms > 0 else 0.0
    bursting = burst_on_s > 0.0 and burst_off_s > 0.0
    burst_cycle_len = burst_on_s + burst_off_s
    cycle_start = now

    end_time = now + duration_s

    try:
        while True:
            now = time.perf_counter()
            if now >= end_time:
                break

            # Determine whether we are in an ON window if bursting is enabled
            in_on_window = True
            if bursting:
                cycle_pos = (now - cycle_start) % burst_cycle_len
                in_on_window = cycle_pos < burst_on_s

            next_due = min(s.next_time for s in states)
            # Sleep until next event (cap to small granularity)
            sleep_for = next_due - now
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.002))
                continue

            for idx, st in enumerate(states):
                if st.next_time - now <= 1e-6:
                    # Compute next value
                    if triangle_mode:
                        nxt_val, st.direction = pattern_triangle(st.current_value, value_min, value_max, max(1, quant_step), st.direction)
                    else:
                        nxt_val = pattern_func(st.current_value, value_min, value_max, max(1, quant_step))  # type: ignore[misc]

                    # Quantize (independent of pattern step). This simulates dedup/thresholding scenarios.
                    q_val = quantize_value(nxt_val, quant_step, value_min, value_max)

                    # Send if allowed and not suppressed by duplicate rule
                    should_send = in_on_window or not bursting
                    if suppress_duplicates and q_val == st.current_value:
                        should_send = False

                    if should_send:
                        send_cc(port, st.cc, st.channel, q_val)
                        sent_counts_total += 1
                        sent_counts_per_stream[idx] += 1
                        if csv_writer is not None:
                            csv_writer.writerow([
                                f"{now - (end_time - duration_s):.6f}",  # relative
                                time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
                                st.cc,
                                st.channel,
                                q_val,
                            ])

                    st.current_value = q_val
                    st.next_time = now + interval_per_cc

            # Stats every ~1s
            if now - last_stats_time >= 1.0:
                per_cc_rate = sent_counts_total / max(1e-9, (now - last_stats_time))
                # Reset windowed counters for per-second rate
                # Instead report totals each second as a rolling metric
                print(f"Rate (approx): {per_cc_rate:.1f} msgs/s (window 1s), Total sent: {sent_counts_total}")
                last_stats_time = now
                sent_counts_total = 0
                # per-stream detailed stats can be added if needed

    finally:
        if csv_file is not None:
            try:
                csv_file.flush()
                csv_file.close()
            except Exception:
                pass


# =============================
# CLI
# =============================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MIDI stress test for Elektron Digitone")
    parser.add_argument("--profile", choices=["current-four", "sweep-csv"], default="current-four", help="Profile of CCs/channels to use")
    parser.add_argument("--csv-path", type=str, default="midi_mapping_digitone.csv", help="CSV path for sweep-csv profile")
    parser.add_argument("--device-index", type=int, default=None, help="MIDI output device index")
    parser.add_argument("--device-name", type=str, default=None, help="Substring to match MIDI output device name")

    parser.add_argument("--rate-total", type=float, default=None, help="Total CC messages per second (divided across streams)")
    parser.add_argument("--rate-per-cc", type=float, default=None, help="CC messages per second per stream (overrides --rate-total)")
    parser.add_argument("--duration", type=float, default=120.0, help="Duration in seconds")
    parser.add_argument("--streams", type=int, default=None, help="Limit number of concurrent CC streams (defaults to all available)")
    parser.add_argument("--channels", type=str, default=None, help="Optional comma-separated MIDI channels to use (0-15)")
    parser.add_argument("--ccs", type=str, default=None, help="Optional comma-separated CC numbers to use (0-127)")

    parser.add_argument("--pattern", choices=["ramp", "triangle", "random", "step"], default="ramp", help="Value pattern per stream")
    parser.add_argument("--quantize-step", type=int, default=1, help="Quantization step size (1=no quantization)")
    parser.add_argument("--suppress-duplicates", action="store_true", help="Do not send when quantized value is unchanged")
    parser.add_argument("--value-min", type=int, default=0, help="Minimum CC value (0-127)")
    parser.add_argument("--value-max", type=int, default=127, help="Maximum CC value (0-127)")

    parser.add_argument("--burst-on-ms", type=int, default=0, help="Burst ON window in ms (0=disabled)")
    parser.add_argument("--burst-off-ms", type=int, default=0, help="Burst OFF window in ms (0=disabled)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for repeatability")

    parser.add_argument("--log-csv", type=str, default=None, help="Path to CSV log file")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Device selection
    outputs = list_midi_outputs()
    print("Available MIDI outputs:")
    for i, out in enumerate(outputs):
        print(f"  [{i}] {out}")

    port = open_midi_output_by_index_or_name(args.device_index, args.device_name)
    if port is None:
        return 2

    # Build pairs
    pairs: List[Tuple[int, int]]
    if args.profile == "current-four":
        pairs = load_profile_current_four()
    else:
        csv_path = Path(args.csv_path)
        chan_list = parse_int_list(args.channels)
        default_channel = chan_list[0] if chan_list else 0
        pairs = load_profile_from_csv(csv_path, default_channel=default_channel, limit=None)

    # Filter by channels/ccs if requested
    cc_filter = parse_int_list(args.ccs)
    if cc_filter:
        pairs = [(cc, ch) for (cc, ch) in pairs if cc in cc_filter]
    chan_filter = parse_int_list(args.channels)
    if chan_filter:
        pairs = [(cc, ch) for (cc, ch) in pairs if ch in chan_filter]

    if not pairs:
        print("No CC/channel pairs to test after filters.")
        return 3

    if args.streams is not None and args.streams > 0:
        pairs = pairs[: args.streams]

    value_min = clamp(args.value_min, 0, 127)
    value_max = clamp(args.value_max, 0, 127)
    if value_min > value_max:
        value_min, value_max = value_max, value_min

    csv_log_path = Path(args.log_csv) if args.log_csv else None

    # Graceful shutdown on SIGINT
    interrupted = {"flag": False}

    def handle_sigint(_: int, __: Optional[object]) -> None:
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        run_stress_test(
            port=port,
            pairs=pairs,
            duration_s=float(args.duration),
            rate_total=args.rate_total,
            rate_per_cc=args.rate_per_cc,
            pattern_name=args.pattern,
            quant_step=int(args.quantize_step),
            suppress_duplicates=bool(args.suppress_duplicates),
            value_min=value_min,
            value_max=value_max,
            burst_on_ms=int(args.burst_on_ms),
            burst_off_ms=int(args.burst_off_ms),
            csv_log_path=csv_log_path,
            seed=args.seed,
        )
        return 0
    finally:
        try:
            port.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())


