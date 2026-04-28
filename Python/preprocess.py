#!/usr/bin/env python3
"""
preprocess.py

Converts raw X_train / Y_train CSV pairs into aligned symbol-level datasets
ready for ML training.

Pipeline (mirrors Receiver.m exactly):
    Y_train (200k I,Q)
        -> SRRC matched filter
        -> Normalize
        -> Eye diagram -> timing offset
        -> Downsample x8 -> complex symbols
        -> PN correlation -> sync -> extract burst
        -> [distorted symbols]    paired with    [clean symbols from X_train]

Output:
    SimulatedData/Processed/
        symbols_rx_<label>.npy   -- complex64, shape (N_syms,)
        symbols_tx_<label>.npy   -- float32,   shape (N_syms,), values in {-1, +1}

    SimulatedData/
        dataset_windowed.npz     -- full windowed ML dataset
            X  : float32 (n_samples, 2*WINDOW)  [I,Q interleaved window]
            y  : float32 (n_samples,)            clean symbol {-1, +1}
            meta: object array of dicts          one per sample (label, sym_idx, etc.)
"""

import numpy as np
import csv
import os
import glob

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
SPS        = 8
N_PN       = 128
WINDOW     = 11          # symbols: 5 past + current + 5 future
HALF_WIN   = WINDOW // 2
DATA_DIR   = r"c:\Users\sohom\Downloads\CSL Proj\SimulatedData"
OUT_DIR    = os.path.join(DATA_DIR, "Processed")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# SIGNAL PROCESSING UTILITIES  (same as generate_dataset.py)
# ─────────────────────────────────────────────────────────────

def srrc_coeffs():
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


def pn_sequence():
    state = [0, 0, 0, 0, 0, 0, 1]
    bits  = []
    for _ in range(N_PN):
        out      = state[6]
        bits.append(out)
        feedback = state[6] ^ state[5]
        state    = [feedback] + state[:6]
    return np.array(bits, dtype=np.float32)


H    = srrc_coeffs()
SYNC = pn_sequence()                   # raw PN, values {0, 1}
# Correlation template: raw PN in bipolar, used AFTER differential decoding (mirrors Receiver.m)
SYNC_BIPOLAR = (2 * SYNC - 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# STEP 1: Clean symbols from X_train
# ─────────────────────────────────────────────────────────────

def clean_symbols_from_x(x_path: str) -> np.ndarray:
    """
    Load X_train, apply SRRC matched filter, downsample at offset 0,
    skip filter ramp-up (4 symbols), return clean bipolar symbols {-1, +1}.
    """
    with open(x_path) as f:
        x = np.array([float(r[0]) for r in csv.reader(f)])

    filtered = np.convolve(H, x.astype(complex))
    norm_val  = max(np.max(np.abs(np.real(filtered))),
                    np.max(np.abs(np.imag(filtered))))
    filtered  = filtered / norm_val

    delay = (len(H) - 1) // 2          # 32 samples = 4 symbols at SPS=8
    syms  = np.real(filtered[delay::SPS])
    syms  = np.sign(syms)              # snap to {-1, +1}

    # Skip the 4-symbol filter ramp-up at each end
    ramp = 4
    return syms[ramp:-ramp].astype(np.float32)


# ─────────────────────────────────────────────────────────────
# STEP 2: Received symbols from Y_train  (mirrors Receiver.m)
# ─────────────────────────────────────────────────────────────

def received_symbols_from_y(y_path: str):
    """
    Full Receiver.m pipeline in Python.

    Returns
    -------
    symbols    : complex ndarray  -- all downsampled symbols (full capture)
    start_idx  : int              -- index of first sync symbol (= start_bit_ind in MATLAB)
    corr       : ndarray          -- full correlation trace (for debug)
    """
    # Load I, Q
    IQ = []
    with open(y_path) as f:
        for row in csv.reader(f):
            IQ.append((float(row[0]), float(row[1])))
    IQ = np.array(IQ)
    rx = IQ[:, 0] + 1j * IQ[:, 1]

    # SRRC matched filter  (Receiver.m: XX_signal = conv(FIR_coeff75, RX_ss))
    xx = np.convolve(H, rx)
    norm_val = max(np.max(np.abs(np.real(xx))), np.max(np.abs(np.imag(xx))))
    rx_sig = xx / norm_val

    # Eye diagram -> find timing offset
    # Use signal POWER (|I|² + |Q|²) instead of I-only so the timing estimate
    # is correct regardless of carrier phase (90° offset, freq offset, etc.).
    eye_len       = SPS
    eye_frame_len = len(rx_sig) // eye_len
    usable        = eye_frame_len * eye_len
    pwr           = (np.abs(rx_sig[:usable]) ** 2).reshape(eye_frame_len, eye_len).T
    eye_var       = np.sum(pwr, axis=1) / eye_frame_len
    eye_offset    = int(np.argmax(eye_var[:SPS])) + 1    # +1: MATLAB is 1-indexed

    # Downsample  (Receiver.m: Symbols(i) = RX_signal((i-1)*8 + eye_offset))
    n_syms  = (len(rx_sig) - eye_offset) // SPS
    indices = eye_offset + np.arange(n_syms) * SPS
    indices = indices[indices < len(rx_sig)]
    symbols = rx_sig[indices]

    # Differential decoding  (mirrors Receiver.m exactly)
    # angle_diff[i] = |angle(symbols[i]) - angle(symbols[i-1])|, with wrap correction
    angles     = np.angle(symbols)
    angle_arr  = np.concatenate([[0.0], angles])
    angle_diff = np.abs(angle_arr[1:] - angle_arr[:-1])
    angle_diff[angle_diff > 3 * np.pi / 2] = 0.0
    bits_bipolar = np.sign(angle_diff - np.pi / 2)   # {-1, +1}

    # PN correlation on differentially decoded bits  (Receiver.m: corval, start_bit_ind)
    n_search = len(bits_bipolar) - N_PN
    corval   = np.array([np.dot(SYNC_BIPOLAR, bits_bipolar[i:i + N_PN])
                         for i in range(n_search)])

    start_idx = int(np.argmax(np.abs(corval)))

    return symbols, start_idx, corval


# ─────────────────────────────────────────────────────────────
# STEP 3: Align and extract burst
# ─────────────────────────────────────────────────────────────

def extract_aligned_pair(x_path: str, y_path: str):
    """
    Returns
    -------
    rx_burst : complex ndarray, shape (N_syms,)  -- distorted received symbols
    tx_clean : float ndarray,  shape (N_syms,)   -- clean reference {-1,+1}
    """
    tx_clean_full = clean_symbols_from_x(x_path)   # 344 symbols (352 - 2*4 ramp)
    n_burst       = len(tx_clean_full)

    symbols, start_idx, _ = received_symbols_from_y(y_path)

    # Extract matching burst from received symbol stream
    end_idx  = start_idx + n_burst
    if end_idx > len(symbols):
        raise ValueError(
            f"Burst extraction out of bounds: start={start_idx}, "
            f"n_burst={n_burst}, total_syms={len(symbols)}"
        )
    rx_burst = symbols[start_idx:end_idx].astype(np.complex64)

    return rx_burst, tx_clean_full


# ─────────────────────────────────────────────────────────────
# STEP 4: Build windowed ML dataset
# ─────────────────────────────────────────────────────────────

def make_windows(rx: np.ndarray, tx: np.ndarray):
    """
    Sliding window over a single burst pair.

    Input feature per sample:  [I_{k-H}, Q_{k-H}, ..., I_k, Q_k, ..., I_{k+H}, Q_{k+H}]
                                shape = (2 * WINDOW,)

    Target: tx[k]  in {-1, +1}

    Returns X (n, 2*WINDOW) and y (n,).
    """
    n      = len(rx)
    X_list = []
    y_list = []
    for k in range(HALF_WIN, n - HALF_WIN):
        window = rx[k - HALF_WIN: k + HALF_WIN + 1]   # complex, length WINDOW
        feat   = np.concatenate([np.real(window), np.imag(window)])
        X_list.append(feat)
        y_list.append(tx[k])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    x_files = sorted(glob.glob(os.path.join(DATA_DIR, "Train", "X_train", "*.csv")))

    all_X, all_y, all_meta = [], [], []
    n_ok, n_fail = 0, 0

    for x_path in x_files:
        label  = os.path.splitext(os.path.basename(x_path))[0]
        y_path = os.path.join(DATA_DIR, "Train", "Y_train", label + ".csv")

        if not os.path.exists(y_path):
            print(f"  SKIP (no Y_train): {label}")
            continue

        try:
            rx_burst, tx_clean = extract_aligned_pair(x_path, y_path)

            # Save per-file symbol arrays
            np.save(os.path.join(OUT_DIR, f"symbols_rx_{label}.npy"), rx_burst)
            np.save(os.path.join(OUT_DIR, f"symbols_tx_{label}.npy"), tx_clean)

            # Build windowed samples for this file
            X_w, y_w = make_windows(rx_burst, tx_clean)
            meta_w   = [{"label": label, "sym_idx": k}
                        for k in range(HALF_WIN, len(rx_burst) - HALF_WIN)]

            all_X.append(X_w)
            all_y.append(y_w)
            all_meta.extend(meta_w)

            n_ok += 1
            print(f"  OK  {label}: {len(rx_burst)} symbols -> {len(X_w)} windows")

        except Exception as e:
            print(f"  FAIL {label}: {e}")
            n_fail += 1

    if all_X:
        X_full = np.concatenate(all_X, axis=0)
        y_full = np.concatenate(all_y, axis=0)

        out_path = os.path.join(DATA_DIR, "dataset_windowed.npz")
        np.savez(out_path, X=X_full, y=y_full, meta=np.array(all_meta, dtype=object))

        print(f"\nDataset saved: {out_path}")
        print(f"  X shape : {X_full.shape}  (n_samples, 2*WINDOW={2*WINDOW})")
        print(f"  y shape : {y_full.shape}")
        print(f"  Classes : {np.unique(y_full)}")
        print(f"  Files OK/FAIL: {n_ok}/{n_fail}")
    else:
        print("No files processed successfully.")


if __name__ == "__main__":
    main()
