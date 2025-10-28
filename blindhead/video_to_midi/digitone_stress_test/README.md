## MIDI Stress Test for Elektron Digitone

A standalone tool to stress test Digitone's MIDI input (via Scarlett 6i6 DIN or other outputs):

Script: `blindhead/video_to_midi/midi_stress_test_digitone.py`

Examples:

- Four-CC steady ramp @ 200 msgs/s total for 2 min:

  ```bash
  python blindhead/video_to_midi/midi_stress_test_digitone.py \
    --profile current-four --rate-total 200 --pattern ramp --duration 120
  ```

- Bursty random jitter, 4 streams, 1000 msgs/s in 100ms bursts each second:

  ```bash
  python blindhead/video_to_midi/midi_stress_test_digitone.py \
    --profile current-four --streams 4 --rate-total 1000 \
    --pattern random --burst-on-ms 100 --burst-off-ms 900 --duration 120
  ```

- Sweep CCs from `midi_mapping_digitone.csv` with quantization to 16 steps @ 300 msgs/s total:

  ```bash
  python blindhead/video_to_midi/midi_stress_test_digitone.py \
    --profile sweep-csv --rate-total 300 --quantize-step 16 \
    --pattern triangle --duration 180
  ```

Flags:
- `--device-index` or `--device-name` to select MIDI output
- `--ccs` and/or `--channels` to filter sets (comma-separated)
- `--log-csv` to capture a CSV trace of sends

### Run the full test suite

You can run all predefined tests sequentially with:

```bash
bash scripts/run_digitone_midi_stress_suite.sh --device-name "Scarlett"
```

Logs are written to `digitone_stress_logs/` with both CSV traces and console outputs, plus a short summary at the end.
