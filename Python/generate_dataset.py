#!/usr/bin/env python3
"""
generate_dataset.py

Generates a simulated BPSK training dataset covering 5 hardware non-idealities.
File format matches real hardware captures in Data.zip exactly, so the same
Receiver.m pipeline and ML preprocessing code works on both.

Signal amplitude and noise floor are calibrated from the real Acquired_data.xlsx
capture. All distortions are applied in software as described in the project brief.

Output:
    SimulatedData/
        dataset_index.csv                     -- master index of all file pairs
        Train/
            X_train/<label>.csv               -- 1-col: clean SRRC-filtered TX signal
            Y_train/<label>.csv               -- 2-col: received I, Q (200k samples)
            Labels/<label>.csv                -- per-run metadata
"""

import numpy as np
import os
import csv

# ─────────────────────────────────────────────────────────────
# SYSTEM PARAMETERS  (match Transmitter.m / Receiver.m exactly)
# ─────────────────────────────────────────────────────────────
SIGNAL_RATE  = 1_600_000      # Hz — DAC and ADC both run at this rate (1.6 MHz)
SPS          = 8              # samples per symbol at SIGNAL_RATE
SYMBOL_RATE  = SIGNAL_RATE // SPS   # = 200 kHz
ROLLOFF      = 0.75
FILTER_SPAN  = 8              # symbols  → filter length = 8*8+1 = 65 taps
N_PN         = 128            # PN sync sequence length
CAPTURE_LEN  = 200_000        # samples in every Y_train file (= 0.125 s at 1.6 MHz)

# Calibrated from real hardware capture (Acquired_data.xlsx / AbCdEfGhIjKlMnO pair)
BASE_AMP  = 0.00097           # peak amplitude of received signal at 100 % TX/RX gain
SIGMA_I   = 0.000675          # noise std on I branch
SIGMA_Q   = 0.000679          # noise std on Q branch

BASE_DIR = r"c:\Users\sohom\Downloads\CSL Proj"
OUT_DIR  = os.path.join(BASE_DIR, "SimulatedData")

# ─────────────────────────────────────────────────────────────
# 5 FIXED MESSAGES  (26 letters, each in a different order)
# ─────────────────────────────────────────────────────────────
MESSAGES = [
    "qwertyuiopasdfghjklzxcvbnm",
    "zyxwvutsrqponmlkjihgfedcba",
    "mnbvcxzlkjhgfdsapoiuytrewq",
    "plokimjunhybgtvfrcdexswzaq",
    "abcdefghijklmnopqrstuvwxyz",
]

# ─────────────────────────────────────────────────────────────
# NON-IDEALITY LEVELS
# ─────────────────────────────────────────────────────────────
SNR_SCALES   = [1.00, 0.75, 0.50, 0.25]      # multiply BASE_AMP
PHASE_DEGS   = [0, 30, 60, 90, 120, 150]     # static phase rotation (degrees)
FREQ_HZ      = [0, 500, 1_000, 2_000, 5_000] # frequency offset (Hz)
IQ_ALPHAS    = [1.00, 0.85, 0.70]            # Q-branch amplitude scale
SALEH_PARAMS = {                              # (alpha_a, beta_a, alpha_p, beta_p)
    "mild":     (2.0, 0.5, 0.5, 1.0),
    "moderate": (2.0, 1.0, 1.5, 1.0),
    "severe":   (2.0, 2.0, 3.0, 1.0),
}
COMBINED = {"phase_deg": 60, "freq_hz": 1_000, "iq_alpha": 0.85}


# ═════════════════════════════════════════════════════════════
# SIGNAL PROCESSING UTILITIES
# ═════════════════════════════════════════════════════════════

def srrc_coeffs():
    """Square Root Raised Cosine filter (matches MATLAB comm.RaisedCosineTransmitFilter)."""
    span, sps, b = FILTER_SPAN, SPS, ROLLOFF
    n   = span * sps + 1
    idx = np.arange(n) - (n - 1) // 2
    t   = idx / sps                         # normalised time in symbol periods
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
    """
    128-bit PN sequence matching MATLAB:
        comm.PNSequence('Polynomial',[7 6 0],'InitialConditions',[0 0 0 0 0 0 1])

    LFSR: degree-7, polynomial x^7+x^6+1, taps at stages 7 and 6.
    State vector [s1..s7], output from s7, feedback = s7 XOR s6.
    """
    state = [0, 0, 0, 0, 0, 0, 1]   # [s1 … s7]
    bits  = []
    for _ in range(N_PN):
        out      = state[6]                   # s7 is the output
        bits.append(out)
        feedback = state[6] ^ state[5]        # taps at s7 and s6
        state    = [feedback] + state[:6]     # shift register
    return np.array(bits, dtype=int)


# Pre-compute shared objects once
SRRC_H = srrc_coeffs()
SYNC   = pn_sequence()


def tx_chain(message: str):
    """
    Full transmitter chain matching Transmitter.m.

    Returns
    -------
    sig_rrc  : ndarray, shape (N_samples,)   normalised SRRC signal  → X_train content
    tx_text  : ndarray, shape (N_bits,)      full bit stream (sync+len+data)
    """
    data_bits = np.unpackbits(
        np.frombuffer(message.encode("ascii"), dtype=np.uint8)
    )
    len_bits = np.array([int(b) for b in format(len(data_bits), "016b")])
    tx_text  = np.concatenate([SYNC, len_bits, data_bits])

    # Differential encoding
    diff    = np.zeros(len(tx_text), dtype=int)
    diff[0] = int(tx_text[0]) ^ 0
    for i in range(1, len(diff)):
        diff[i] = diff[i - 1] ^ int(tx_text[i])

    # Bipolar mapping → upsample → SRRC filter
    bpolar    = 2.0 * diff - 1.0
    upsampled = np.zeros(len(bpolar) * SPS)
    upsampled[::SPS] = bpolar
    sig = np.convolve(SRRC_H, upsampled)
    sig = sig / np.max(np.abs(sig))           # normalise to ±1  (matches MATLAB)
    return sig, tx_text


def saleh_distort(sig_real: np.ndarray, aa, ba, ap, bp) -> np.ndarray:
    """
    Saleh PA model (TX-side distortion, applied before embedding in noise).
    AM-AM: A_out = (alpha_a * A_in) / (1 + beta_a * A_in^2)
    AM-PM: phi   = (alpha_p * A_in^2) / (1 + beta_p * A_in^2)
    Returns complex distorted signal.
    """
    s   = sig_real.astype(complex)
    A   = np.abs(s)
    A_o = (aa * A) / (1 + ba * A ** 2)
    phi = (ap * A ** 2) / (1 + bp * A ** 2)
    return A_o * np.exp(1j * (np.angle(s) + phi))


def embed(sig_complex: np.ndarray, rng: np.random.Generator,
          freq_offset_hz: float = 0.0) -> tuple:
    """
    Tile the burst to fill CAPTURE_LEN samples (matching real hardware behaviour),
    optionally apply a continuous frequency offset across the full capture,
    then add AWGN.

    Frequency offset is applied AFTER tiling so the phase accumulates
    continuously across burst repetitions — matching real hardware behaviour
    where the oscillator drift is independent of the message framing.
    """
    n_tiles = int(np.ceil(CAPTURE_LEN / len(sig_complex)))
    tiled   = np.tile(sig_complex, n_tiles)[:CAPTURE_LEN]

    if freq_offset_hz != 0.0:
        t_full = np.arange(CAPTURE_LEN) / SIGNAL_RATE
        tiled  = tiled * np.exp(1j * 2 * np.pi * freq_offset_hz * t_full)

    I = np.real(tiled) + rng.normal(0, SIGMA_I, CAPTURE_LEN)
    Q = np.imag(tiled) + rng.normal(0, SIGMA_Q, CAPTURE_LEN)
    return I, Q


# ═════════════════════════════════════════════════════════════
# FILE I/O
# ═════════════════════════════════════════════════════════════

def write_x(path: str, data: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for v in data:
            w.writerow([f"{v:.8f}"])


def write_y(path: str, I: np.ndarray, Q: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for i, q in zip(I, Q):
            w.writerow([f"{i:.8f}", f"{q:.8f}"])


def write_label(path: str, info: dict):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for k, v in info.items():
            w.writerow([k, v])


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    # Create output directories
    for sub in ["Train/X_train", "Train/Y_train", "Train/Labels"]:
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

    index = [["filename", "split", "non_ideality", "param_name", "param_value", "message_id"]]
    total = 0

    print(f"Calibration: BASE_AMP={BASE_AMP:.5f}  sigma_I={SIGMA_I:.6f}  sigma_Q={SIGMA_Q:.6f}")
    print(f"Signal rate: {SIGNAL_RATE/1e6:.1f} MHz  Symbol rate: {SYMBOL_RATE/1e3:.0f} kHz  SPS={SPS}")
    print(f"Burst tiled (no decimation) across {CAPTURE_LEN}-sample capture.\n")

    for mid, message in enumerate(MESSAGES, 1):
        sig_clean, tx_text = tx_chain(message)
        sig_len = len(sig_clean)

        # Independent noise seed per message so runs differ
        rng = np.random.default_rng(seed=1000 * mid)

        print(f"Message {mid}: '{message}'  ({sig_len} samples in X_train)")

        def save(label, sig_c, ni, pname, pval, extra_meta=None, freq_hz=0.0):
            """Scale sig_c by BASE_AMP, embed in noise, write X/Y/Label files."""
            nonlocal total
            xp = os.path.join(OUT_DIR, "Train", "X_train", f"{label}.csv")
            yp = os.path.join(OUT_DIR, "Train", "Y_train", f"{label}.csv")
            lp = os.path.join(OUT_DIR, "Train", "Labels",  f"{label}.csv")

            I, Q = embed(sig_c * BASE_AMP, rng, freq_offset_hz=freq_hz)
            write_x(xp, sig_clean)
            write_y(yp, I, Q)

            meta = {
                "non_ideality":  ni,
                "param_name":    pname,
                "param_value":   pval,
                "message_id":    mid,
                "message":       message,
                "base_amp":      BASE_AMP,
                "sigma_i":       SIGMA_I,
                "sigma_q":       SIGMA_Q,
            }
            if extra_meta:
                meta.update(extra_meta)
            write_label(lp, meta)
            index.append(["Train/" + label, "Train", ni, pname, pval, mid])
            total += 1

        # ── 1. TX Power / RX Gain (SNR sweep) ─────────────────────────
        for scale in SNR_SCALES:
            lbl = f"snr_{int(scale * 100):03d}pct_msg{mid}"
            sig_c = sig_clean.astype(complex) * scale
            save(lbl, sig_c, "snr", "amplitude_scale", scale)

        # ── 2. Phase Offset ───────────────────────────────────────────
        for deg in PHASE_DEGS:
            lbl = f"phase_{deg:03d}deg_msg{mid}"
            sig_c = sig_clean.astype(complex) * np.exp(1j * np.deg2rad(deg))
            save(lbl, sig_c, "phase_offset", "degrees", deg)

        # ── 3. Frequency Offset ───────────────────────────────────────
        # Applied AFTER tiling via embed() so phase accumulates continuously
        # across all burst repetitions — matches real hardware oscillator drift.
        for df in FREQ_HZ:
            lbl = f"freq_{df:05d}hz_msg{mid}"
            sig_c = sig_clean.astype(complex)        # no pre-rotation
            save(lbl, sig_c, "freq_offset", "hz", df, freq_hz=float(df))

        # ── 4. IQ Imbalance ───────────────────────────────────────────
        # Applied AFTER noise addition (hardware reality: mixer imbalance at RX)
        for alpha in IQ_ALPHAS:
            lbl = f"iq_{int(alpha * 100):03d}_msg{mid}"
            xp = os.path.join(OUT_DIR, "Train", "X_train", f"{lbl}.csv")
            yp = os.path.join(OUT_DIR, "Train", "Y_train", f"{lbl}.csv")
            lp = os.path.join(OUT_DIR, "Train", "Labels",  f"{lbl}.csv")

            sig_c = sig_clean.astype(complex) * BASE_AMP
            I, Q  = embed(sig_c, rng)
            Q     = alpha * Q           # scale Q branch after embedding (incl. noise)

            write_x(xp, sig_clean)
            write_y(yp, I, Q)
            write_label(lp, {
                "non_ideality": "iq_imbalance", "param_name": "q_scale",
                "param_value": alpha, "message_id": mid, "message": message,
                "base_amp": BASE_AMP, "sigma_i": SIGMA_I, "sigma_q": SIGMA_Q,
            })
            index.append(["Train/" + lbl, "Train", "iq_imbalance", "q_scale", alpha, mid])
            total += 1

        # ── 5. Nonlinear Distortion (Saleh model, TX side) ───────────
        for severity, (aa, ba, ap, bp) in SALEH_PARAMS.items():
            lbl = f"saleh_{severity}_msg{mid}"
            # Distort clean signal → re-normalise → scale by BASE_AMP inside save()
            distorted_c = saleh_distort(sig_clean, aa, ba, ap, bp)
            distorted_c = distorted_c / np.max(np.abs(distorted_c))
            save(lbl, distorted_c, "nonlinear_saleh", "severity", severity,
                 extra_meta={"saleh_alpha_a": aa, "saleh_beta_a": ba,
                             "saleh_alpha_p": ap, "saleh_beta_p": bp})

        # ── 6. Combined (phase + frequency offset + IQ imbalance) ─────
        lbl = f"combined_msg{mid}"
        xp  = os.path.join(OUT_DIR, "Train", "X_train", f"{lbl}.csv")
        yp  = os.path.join(OUT_DIR, "Train", "Y_train", f"{lbl}.csv")
        lp  = os.path.join(OUT_DIR, "Train", "Labels",  f"{lbl}.csv")

        sig_c = sig_clean.astype(complex)
        sig_c = sig_c * np.exp(1j * np.deg2rad(COMBINED["phase_deg"]))
        # freq offset applied continuously in embed(); IQ imbalance after embed()
        sig_c = sig_c * BASE_AMP
        I, Q  = embed(sig_c, rng, freq_offset_hz=float(COMBINED["freq_hz"]))
        Q     = COMBINED["iq_alpha"] * Q        # IQ imbalance after embedding

        write_x(xp, sig_clean)
        write_y(yp, I, Q)
        write_label(lp, {
            "non_ideality": "combined",
            "phase_deg":    COMBINED["phase_deg"],
            "freq_hz":      COMBINED["freq_hz"],
            "iq_alpha":     COMBINED["iq_alpha"],
            "message_id":   mid, "message": message,
            "base_amp":     BASE_AMP,
        })
        index.append(["Train/" + lbl, "Train", "combined",
                       f"phase+freq+iq",
                       f"{COMBINED['phase_deg']}deg+{COMBINED['freq_hz']}hz+{COMBINED['iq_alpha']}iq",
                       mid])
        total += 1

    # Write master index
    idx_path = os.path.join(OUT_DIR, "dataset_index.csv")
    with open(idx_path, "w", newline="") as f:
        csv.writer(f).writerows(index)

    expected = (len(SNR_SCALES) + len(PHASE_DEGS) + len(FREQ_HZ)
                + len(IQ_ALPHAS) + len(SALEH_PARAMS) + 1) * len(MESSAGES)
    print(f"\nDone — {total} file pairs written to {OUT_DIR}")
    print(f"Expected: ({len(SNR_SCALES)}+{len(PHASE_DEGS)}+{len(FREQ_HZ)}"
          f"+{len(IQ_ALPHAS)}+{len(SALEH_PARAMS)}+1) × {len(MESSAGES)} = {expected}")
    print(f"Dataset index: {idx_path}")


if __name__ == "__main__":
    main()
