#!/usr/bin/env bash

set -u -o pipefail

# Run a suite of MIDI stress tests against Elektron Digitone
# Requires: python3, mido + rtmidi backend, and the stress script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON=${PYTHON:-python3}
STRESS_PY="${REPO_ROOT}/blindhead/video_to_midi/digitone_stress_test/midi_stress_test_digitone.py"
OUT_DIR="${REPO_ROOT}/digitone_stress_logs"

DEVICE_NAME=""
DEVICE_INDEX=""
CSV_PATH="${REPO_ROOT}/blindhead/video_to_midi/midi_mapping_digitone.csv"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--device-name NAME | --device-index N] [--csv-path PATH]

Examples:
  $(basename "$0") --device-name "Scarlett"
  $(basename "$0") --device-index 0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device-name)
      DEVICE_NAME="$2"; shift 2;;
    --device-index)
      DEVICE_INDEX="$2"; shift 2;;
    --csv-path)
      CSV_PATH="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ ! -f "$STRESS_PY" ]]; then
  echo "Stress script not found: $STRESS_PY" >&2
  exit 3
fi

mkdir -p "$OUT_DIR"

select_device_args=()
if [[ -n "$DEVICE_NAME" ]]; then
  select_device_args+=(--device-name "$DEVICE_NAME")
elif [[ -n "$DEVICE_INDEX" ]]; then
  select_device_args+=(--device-index "$DEVICE_INDEX")
else
  echo "No device specified; will use the first available MIDI output."
fi

timestamp() { date +"%Y-%m-%dT%H:%M:%S"; }

run_test() {
  local name="$1"; shift
  local csv_log="$OUT_DIR/${name}.csv"
  local out_log="$OUT_DIR/${name}.out"

  echo "["$(timestamp)"] Starting: $name" | tee -a "$out_log"
  set +e
  "$PYTHON" "$STRESS_PY" "$@" --log-csv "$csv_log" "${select_device_args[@]}" 2>&1 | tee -a "$out_log"
  local rc=${PIPESTATUS[0]}
  set -e

  # Basic metrics
  local sent=0
  if [[ -f "$csv_log" ]]; then
    sent=$(( $(wc -l < "$csv_log") - 1 ))
    if (( sent < 0 )); then sent=0; fi
  fi
  local duration=$(grep -Eo "--duration [0-9]+(\.[0-9]+)?" <<<"$*" | awk '{print $2}')
  if [[ -z "$duration" ]]; then duration=0; fi
  local rate=0
  if (( $(printf '%.0f' "$duration") > 0 )); then
    rate=$(python3 - <<PY
import sys
sent=${sent}
dur=float("${duration}")
print(f"{(sent/dur):.1f}")
PY
)
  fi

  if [[ $rc -eq 0 ]]; then
    echo "["$(timestamp)"] Finished: $name (OK) sent=${sent} est_rate=${rate} msgs/s" | tee -a "$out_log"
  else
    echo "["$(timestamp)"] Finished: $name (ERROR rc=${rc}) sent=${sent} est_rate=${rate} msgs/s" | tee -a "$out_log"
  fi
}

# Test set
set +e

# 1) Steady ramp: 100, 300, 600, 1000 msgs/s, 120s
# run_test test01_ramp_100  --profile current-four --rate-total 100  --pattern ramp     --duration 120
# run_test test02_ramp_300  --profile current-four --rate-total 300  --pattern ramp     --duration 120

# AlexK: one crash here (Exception DN0097 V03 M0 P4012C36A)
# run_test test03_ramp_600  --profile current-four --rate-total 600  --pattern ramp     --duration 120

# AlexK: sending 1000 msgs/s leads to Exception DN0097 V03 M0 P4012C36A after a few seconds max reliably 
# run_test test04_ramp_1000 --profile current-four --rate-total 1000 --pattern ramp     --duration 120

# 2) Quantization impact (triangle, step=8, suppress duplicates), 600 msgs/s, 120s
# AlexK: no problem here
# run_test test05_tri_q8_600 --profile current-four --rate-total 600 --pattern triangle --quantize-step 8 --suppress-duplicates --duration 120

# 3) Bursty random 2000 msgs/s with 100ms on / 900ms off, 300s
# AlexK: no problem here
# run_test test06_burst_rand_2k --profile current-four --rate-total 2000 --pattern random --burst-on-ms 100 --burst-off-ms 900 --duration 300

# 4) Sweep CSV (triangle, q=16), 300 msgs/s, 180s
run_test test07_sweepcsv_tri_q16 --profile sweep-csv --csv-path "$CSV_PATH" --rate-total 300 --pattern triangle --quantize-step 16 --suppress-duplicates --duration 180

# 5) Edge case toggling (step) 1000 msgs/s, 120s
run_test test08_step_1000 --profile current-four --rate-total 1000 --pattern step --duration 120

echo
echo "Suite complete. Logs in: $OUT_DIR"
echo "Summaries:"
for f in "$OUT_DIR"/*.out; do
  echo "--- $(basename "$f") ---"
  tail -n 3 "$f" || true
done

exit 0


