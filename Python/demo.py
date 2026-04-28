#!/usr/bin/env python3
"""
demo.py  --  One-command DBPSK demo (Python-only pipeline)
===========================================================
Usage:
    python demo.py <path_to_rx_csv>

If no argument is given, you will be prompted to enter the file path.

What it does:
    1. SRRC matched filter + timing recovery + AFC (if needed)
    2. PN correlation -> find burst start, classical text decode
    3. ML decoding (all trained models)
    4. Print comparison table

Requires: receiver_sim.py and demo_live.py in the same folder.
"""

import sys
import os
import csv
import subprocess
import importlib.util

# Force stdout to flush immediately so headers appear before subprocess output
sys.stdout.reconfigure(line_buffering=True)

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Get RX file path ──────────────────────────────────────────
if len(sys.argv) >= 2:
    rx_file = sys.argv[1]
else:
    print()
    print("  Drag-and-drop the RX CSV file here, or type the full path.")
    rx_file = input("  RX file: ").strip().strip('"').strip("'")

if not rx_file:
    sys.exit("No file provided.")

if not os.path.isfile(rx_file):
    sys.exit(f"ERROR: file not found:\n  {rx_file}")

# ── Step 1: Python receiver ───────────────────────────────────
print()
print("=" * 62)
print("  STEP 1 -- Receiver  (SRRC filter, timing, AFC, PN sync)")
print("=" * 62)
sys.stdout.flush()

r1 = subprocess.run(
    [sys.executable, os.path.join(ROOT, "receiver_sim.py"), rx_file]
)
if r1.returncode != 0:
    sys.exit("Receiver step failed -- see errors above.")

# ── Inject reference message into demo_meta.txt (if available) ──
# Labels live in a sibling folder called Labels/ next to Y_train/ or Y_test/.
ref_msg = None
rx_abs  = os.path.abspath(rx_file)
label_path = os.path.join(
    os.path.dirname(os.path.dirname(rx_abs)),   # parent of Y_*/
    'Labels',
    os.path.basename(rx_abs),
)
if os.path.isfile(label_path):
    with open(label_path, newline='') as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() == 'message':
                ref_msg = row[1].strip()
                break

if ref_msg:
    meta_path = os.path.join(ROOT, 'demo_meta.txt')
    with open(meta_path, 'a') as f:
        f.write(f'reference={ref_msg}\n')

# ── Step 2: ML decoding (imported directly -- no subprocess) ──
print()
print("=" * 62)
print("  STEP 2 -- ML Decoding  (all trained models)")
print("=" * 62)
sys.stdout.flush()

# Load and call demo_live.main() in-process so it cannot run twice
_dl_path = os.path.join(ROOT, "demo_live.py")
_spec = importlib.util.spec_from_file_location("demo_live", _dl_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_mod.main()
