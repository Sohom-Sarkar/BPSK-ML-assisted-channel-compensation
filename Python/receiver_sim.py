#!/usr/bin/env python3
"""
receiver_sim.py
===============
Python receiver for the hardware-capture CSV files (Y_test format).
  - 200,000 I/Q samples at 1.6 MHz (8 samples/symbol)
  - May contain a carrier frequency offset (hardware drift)

Pipeline:
  1. SRRC matched filter + downsample to symbols
  2. Two-stage AFC to remove carrier offset (coarse +/-15 kHz, fine +/-500 Hz)
  3. PN correlation to find burst start
  4. Decode classical text
  5. Export demo_symbols.csv + demo_meta.txt for demo_live.py

Usage:
    python receiver_sim.py <path_to_rx_csv>

Example:
    python receiver_sim.py "Test (1)/Test/Y_test/TjFcUbSgAeIwOkV_rx.csv"
"""

import sys
import os
import numpy as np
import csv

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Signal parameters ─────────────────────────────────────────
SPS         = 8
SIGNAL_RATE = 1_600_000       # ADC sample rate (Hz)
SYMBOL_RATE = SIGNAL_RATE // SPS  # 200 kHz
N_PN        = 128
DATA_SKIP   = 144             # PN(128) + len_field(16)

# ── SRRC filter (65 taps, span=8, SPS=8, roll-off=0.75) ──────
def _srrc():
    span, sps, b = 8, SPS, 0.75
    n   = span * sps + 1
    idx = np.arange(n) - (n - 1) // 2
    t   = idx / sps
    h   = np.zeros(n)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - b + 4.0 * b / np.pi
        elif abs(abs(ti) - 1.0 / (4.0 * b)) < 1e-9:
            h[i] = (b / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * b)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * b)))
        else:
            num = (np.sin(np.pi * ti * (1 - b))
                   + 4 * b * ti * np.cos(np.pi * ti * (1 + b)))
            den = np.pi * ti * (1 - (4 * b * ti) ** 2)
            h[i] = num / den
    return h

H = _srrc()

# ── PN sequence (x^7+x^6+1, init=[0,0,0,0,0,0,1]) ───────────
def _pn128():
    state = [0, 0, 0, 0, 0, 0, 1]
    bits  = []
    for _ in range(N_PN):
        out  = state[6]
        bits.append(out)
        fb   = state[6] ^ state[5]
        state = [fb] + state[:6]
    return np.array(bits, dtype=np.float32)

SYNC_BIPOLAR = (2 * _pn128() - 1)   # {-1, +1}

# ── Receiver front-end ────────────────────────────────────────
def to_symbols(rx_iq: np.ndarray) -> np.ndarray:
    """SRRC matched filter + eye-diagram timing recovery + x8 downsample."""
    xx     = np.convolve(H, rx_iq)
    nv     = max(np.max(np.abs(xx.real)), np.max(np.abs(xx.imag)))
    rx_sig = xx / nv

    n_frames = len(rx_sig) // SPS
    usable   = n_frames * SPS
    pwr      = (np.abs(rx_sig[:usable]) ** 2).reshape(n_frames, SPS).T
    eye_var  = pwr.sum(axis=1) / n_frames
    offset   = int(np.argmax(eye_var[:SPS])) + 1   # +1: MATLAB 1-indexed

    n_syms  = (len(rx_sig) - offset) // SPS
    indices = offset + np.arange(n_syms) * SPS
    return rx_sig[indices[indices < len(rx_sig)]]


def diff_bits(symbols: np.ndarray) -> np.ndarray:
    """Differential phase decode -> bipolar {-1,+1}."""
    angles = np.angle(symbols)
    arr    = np.concatenate([[0.0], angles])
    adiff  = np.abs(arr[1:] - arr[:-1])
    adiff[adiff > 3 * np.pi / 2] = 0.0
    return np.sign(adiff - np.pi / 2)


def pn_correlate(bits_bp: np.ndarray):
    """Cross-correlate SYNC_BIPOLAR with bits_bp. Returns (peak, index)."""
    n = len(bits_bp) - N_PN
    if n <= 0:
        return 0.0, 0
    corval = np.array([np.dot(SYNC_BIPOLAR, bits_bp[i: i + N_PN])
                       for i in range(n)])
    idx = int(np.argmax(np.abs(corval)))
    return float(np.abs(corval[idx])), idx, corval


# ── Two-stage AFC ─────────────────────────────────────────────
def _squared_spectrum_score(symbols: np.ndarray, f_hz: float) -> float:
    """|mean(s_comp^2)| metric -- maximised when freq offset is corrected."""
    k   = np.arange(len(symbols), dtype=np.float64)
    sc  = symbols * np.exp(-1j * 2.0 * np.pi * f_hz * k / SYMBOL_RATE)
    return float(np.abs(np.mean(sc.astype(np.complex128) ** 2)))


def afc(symbols: np.ndarray):
    """
    Two-stage automatic frequency correction.
    Stage 1: coarse grid [-15, +15] kHz in 300 Hz steps
    Stage 2: fine grid [best +/- 500] Hz in 25 Hz steps
    Returns (f_best_hz, compensated_symbols).
    """
    # Coarse
    cands  = np.arange(-15_000, 15_001, 300, dtype=float)
    scores = [_squared_spectrum_score(symbols, f) for f in cands]
    f_c    = float(cands[np.argmax(scores)])
    print(f"  [Coarse AFC] best = {f_c:+.0f} Hz  score = {max(scores):.5f}")

    # Fine
    cands2  = np.arange(f_c - 500, f_c + 501, 25, dtype=float)
    scores2 = [_squared_spectrum_score(symbols, f) for f in cands2]
    f_best  = float(cands2[np.argmax(scores2)])

    k      = np.arange(len(symbols), dtype=np.float64)
    syms_c = (symbols * np.exp(-1j * 2.0 * np.pi * f_best * k / SYMBOL_RATE)
              ).astype(np.complex64)

    peak_c, _, _ = pn_correlate(diff_bits(syms_c))
    print(f"  [Fine  AFC]  best = {f_best:+.0f} Hz  PN peak after AFC = {peak_c:.1f}/{N_PN}")
    return f_best, syms_c


# ── Text decoder ──────────────────────────────────────────────
def decode_text(bits_rec: np.ndarray, start: int):
    """Returns (text, n_chars) where n_chars=0 on failure."""
    len_bits = bits_rec[start + N_PN : start + DATA_SKIP]
    sLen     = ''.join(map(str, len_bits))
    msg_len  = int(sLen, 2)
    if not (8 <= msg_len <= 2048 and msg_len % 8 == 0):
        return f'[bad length field: {msg_len}]', 0
    data_bits = bits_rec[start + DATA_SKIP : start + DATA_SKIP + msg_len]
    chars = []
    for i in range(msg_len // 8):
        byte = data_bits[i * 8 : i * 8 + 8]
        try:
            c = chr(int(''.join(map(str, byte)), 2))
            chars.append(c if c.isprintable() else '?')
        except Exception:
            chars.append('?')
    return ''.join(chars), msg_len // 8


# ── Main ─────────────────────────────────────────────────────
if len(sys.argv) < 2:
    sys.exit("Usage: python receiver_sim.py <rx_csv_file>")

rx_file = sys.argv[1]
if not os.path.exists(rx_file):
    sys.exit(f"ERROR: file not found: {rx_file}")

# 1. Load
rows = []
with open(rx_file) as f:
    for row in csv.reader(f):
        rows.append((float(row[0]), float(row[1])))
a   = np.array(rows)
raw = (a[:, 0] + 1j * a[:, 1]).astype(np.complex64)
print(f"Loaded {len(raw):,} samples from {os.path.basename(rx_file)}")

# 2. SRRC + downsample
symbols = to_symbols(raw)
print(f"Symbols after downsample: {len(symbols):,}")

# 3. Quick PN check (no AFC)
bits_raw          = diff_bits(symbols)
peak_raw, _, _    = pn_correlate(bits_raw)
print(f"PN peak (no AFC): {peak_raw:.1f}/{N_PN}  "
      f"({'OK' if peak_raw > 80 else 'low -- running AFC'})")

# 4. AFC if needed
if peak_raw <= 80:
    print("Running two-stage AFC...")
    f_afc, symbols = afc(symbols)
else:
    f_afc = 0.0

# 5. PN correlation on corrected symbols
bits_c           = diff_bits(symbols)
peak, start, corval = pn_correlate(bits_c)
ss               = np.sign(corval[start])
print(f"PN peak (final): {peak:.1f}/{N_PN}  start={start}  ss={ss:+.0f}")

if peak < 70:
    print("WARNING: PN peak below 70 -- decoded text may be unreliable.")

# 6. Decode text
bits_rec        = ((ss * bits_c + 1) / 2).astype(int)
text, n_chars   = decode_text(bits_rec, start)
print(f"Classical decoded: '{text}'")

# 7. Export for demo_live.py (full burst from PN start, 360 symbols)
n_save    = min(360, len(symbols) - start)
if n_save < 200:
    print("WARNING: not enough symbols for ML export.")
    sys.exit(1)

sym_slice = symbols[start : start + n_save]
out_sym   = os.path.join(ROOT, 'demo_symbols.csv')
out_meta  = os.path.join(ROOT, 'demo_meta.txt')

np.savetxt(out_sym, np.column_stack([sym_slice.real, sym_slice.imag]),
           delimiter=',', fmt='%.10f')

with open(out_meta, 'w') as f:
    f.write(f'peak={round(peak)}\n')
    f.write(f'start={start}\n')
    f.write(f'classical={text}\n')
    if n_chars > 0:
        f.write(f'n_chars={n_chars}\n')

print()
print("--- Export done ---")
print(f"Saved {n_save} symbols -> demo_symbols.csv")
print("Now run: python demo_live.py")
