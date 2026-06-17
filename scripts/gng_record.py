"""
Records GNG_Markers and PiEEG streams to CSV files.

Run this in a separate terminal alongside gng_task.py:

    python scripts/gng_record.py
    python scripts/gng_record.py --out-dir data/gng --subject S01

Outputs
-------
  <out_dir>/<prefix>_markers_<datetime>.csv   — LSL timestamp + marker code + label
  <out_dir>/<prefix>_eeg_<datetime>.csv       — LSL timestamp + ch0 … chN

Recording stops automatically when the MARKER_END (99) marker arrives,
or immediately on Ctrl-C.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import IO, Any

from pylsl import StreamInlet, resolve_byprop

MARKER_END = 99

MARKER_LABELS = {
    5:  "S1_warning",
    10: "go_onset",
    20: "nogo_onset",
    30: "button_press",
    11: "hit",
    12: "miss",
    21: "correct_rejection",
    22: "false_alarm",
    99: "end",
}

DEFAULTS = dict(
    eeg_stream="PiEEG",
    marker_stream="GNG_Markers",
    out_dir=".",
    prefix="gng",
    timeout=30.0,
)


def parse_args():
    p = argparse.ArgumentParser(description="Record GNG markers and EEG to CSV")
    p.add_argument("--eeg-stream",    default=DEFAULTS["eeg_stream"])
    p.add_argument("--marker-stream", default=DEFAULTS["marker_stream"])
    p.add_argument("--out-dir",       default=DEFAULTS["out_dir"],
                   help="Directory for output CSV files")
    p.add_argument("--prefix",        default=DEFAULTS["prefix"],
                   help="Filename prefix (e.g. subject ID)")
    p.add_argument("--timeout",       type=float, default=DEFAULTS["timeout"],
                   help="LSL stream search timeout in seconds")
    p.add_argument("--no-eeg",        action="store_true",
                   help="Skip EEG stream (markers only)")
    return p.parse_args()


def connect(stream_name: str, timeout: float) -> StreamInlet | None:
    print(f"[LSL] Searching for '{stream_name}' …", flush=True)
    streams = resolve_byprop("name", stream_name, timeout=timeout)
    if not streams:
        print(f"[LSL] '{stream_name}' not found.")
        return None
    inlet = StreamInlet(streams[0], max_buflen=360)
    info = streams[0]
    print(f"[LSL] Connected to '{stream_name}'  "
          f"({info.channel_count()} ch, {info.nominal_srate():.0f} Hz)", flush=True)
    return inlet


def open_csv(path: str, header: list[str]) -> tuple[Any, IO[Any]]:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fh = open(path, "w", newline="")
    w = csv.writer(fh)
    w.writerow(header)
    return w, fh


def main():
    args = parse_args()

    marker_inlet = connect(args.marker_stream, args.timeout)
    if marker_inlet is None:
        print("ERROR: GNG_Markers stream not found. Is gng_task.py running?")
        sys.exit(1)

    eeg_inlet = None
    eeg_n_ch  = 0
    if not args.no_eeg:
        eeg_inlet = connect(args.eeg_stream, args.timeout)
        if eeg_inlet is None:
            print("[LSL] EEG stream not found — recording markers only.")
        else:
            eeg_n_ch = eeg_inlet.info().channel_count()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    marker_path = os.path.join(args.out_dir, f"{args.prefix}_markers_{stamp}.csv")
    eeg_path    = os.path.join(args.out_dir, f"{args.prefix}_eeg_{stamp}.csv")

    m_writer, m_fh = open_csv(marker_path, ["lsl_timestamp", "marker_code", "marker_label", "trial_type"])
    print(f"[REC] Markers → {marker_path}")

    e_writer: Any = None
    e_fh: IO[Any] | None = None
    if eeg_inlet is not None:
        ch_header = ["lsl_timestamp"] + [f"ch{i}" for i in range(eeg_n_ch)]
        e_writer, e_fh = open_csv(eeg_path, ch_header)
        print(f"[REC] EEG     → {eeg_path}")

    print("\n[REC] Recording … (Ctrl-C to stop early)\n", flush=True)

    n_markers   = 0
    n_eeg       = 0
    done        = False
    trial_type  = ""   # updated on each go/nogo onset marker

    try:
        while not done:
            # ── drain markers ────────────────────────────────────────────
            while True:
                sample, ts = marker_inlet.pull_sample(timeout=0.0)
                if sample is None:
                    break
                code  = int(sample[0])
                label = MARKER_LABELS.get(code, f"unknown_{code}")
                if code == 10:
                    trial_type = "go"
                elif code == 20:
                    trial_type = "nogo"
                m_writer.writerow([f"{ts:.6f}", code, label, trial_type])
                n_markers += 1
                print(f"  [MARKER] {ts:.3f}  {code:3d}  {label:<20s}  {trial_type}")
                if code == MARKER_END:
                    done = True

            # ── drain EEG ────────────────────────────────────────────────
            if eeg_inlet is not None and e_writer is not None:
                chunk, timestamps = eeg_inlet.pull_chunk(timeout=0.0, max_samples=256)
                if chunk:
                    for ts, sample in zip(timestamps, chunk):
                        e_writer.writerow([f"{ts:.6f}"] + [f"{v:.6f}" for v in sample])
                    n_eeg += len(timestamps)

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[REC] Interrupted by user.")

    finally:
        m_fh.flush(); m_fh.close()
        if e_fh:
            e_fh.flush(); e_fh.close()

    print(f"\n[REC] Done.  {n_markers} markers, {n_eeg} EEG samples saved.")
    print(f"      {marker_path}")
    if e_fh:
        print(f"      {eeg_path}")


if __name__ == "__main__":
    main()
