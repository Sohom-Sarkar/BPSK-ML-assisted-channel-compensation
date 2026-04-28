#!/usr/bin/env python3
"""
evaluate_hardware.py

Evaluates all trained ML equalizers on real hardware captures from Test(1).zip.
Implements a full two-stage AFC (Automatic Frequency Correction) loop to
compensate the ~9 kHz carrier offset present in the hardware captures.

AFC Pipeline per file
---------------------
  1. Load Y_test  (200 k I/Q samples at 1.6 MHz ADC rate)
  2. SRRC matched filter -> normalize -> eye-diagram timing -> downsample to symbols
  3. Coarse AFC:  grid search [-15, +15] kHz in 300 Hz steps (101 candidates)
     Metric: max |PN cross-correlation| on differentially decoded compensated bits
  4. Fine AFC:    grid search [f_coarse +/- 500] Hz in 25 Hz steps (41 candidates)
  5. Apply best f_afc to symbol stream: s_c[k] = s[k].exp(-j.2pi.f.k / f_sym)
  6. Differential decode compensated symbols -> PN correlation -> start_idx
  7. Extract data burst: s_c[start_idx + 144 : start_idx + 348]
     (skip 128 PN + 16 length-header symbols; 204 data symbols remain)
  8. TX reference: clean_symbols_from_x(X_Test) -> 204 diff-encoded symbols
                   orig_bits()                  -> 204 original information bits
  9. Evaluate all 8 models; record BER per file
 10. Save hardware_ber_results.csv and ber_results_combined.csv

Refs: Proakis & Salehi, "Digital Communications", 5th ed. (2008);
      Honkala et al., IEEE TWC 2021 (DeepRx);
      Caciularu & Burshtein, IEEE Commun. Lett. 2020 (CVNN blind eq.)
"""

import numpy as np
import os
import csv
import pickle
import torch

from sklearn.preprocessing import StandardScaler

from common import (
    WINDOW, HALF_WIN, MODEL_DIR, RESULT_DIR,
    make_features_iq, make_features_phase, make_features_combined,
    MLP, ResNetMLP, CVNN, LSTMEqualizer,
    orig_bits,
)

# -------------------------------------------------------------
# HARDWARE / SIGNAL CONSTANTS
# -------------------------------------------------------------
SPS          = 8
SIGNAL_RATE  = 1_600_000          # ADC/DAC sample rate (Hz)
SYMBOL_RATE  = SIGNAL_RATE // SPS  # 200 kHz
N_PN         = 128                # length of PN sync sequence
HEADER_LEN   = 16                 # length-field bits after PN
DATA_SKIP    = N_PN + HEADER_LEN  # 144 symbols to skip past PN+header
HW_DATA_SYMS = 204                # data symbols per hardware burst (after SRRC ramp)

HW_DIR    = os.path.join(os.path.dirname(__file__), "HardwareTest", "Test")
X_DIR     = os.path.join(HW_DIR, "X_Test")
Y_DIR     = os.path.join(HW_DIR, "Y_test")

# AFC search parameters
COARSE_F_MIN  = -15_000  # Hz
COARSE_F_MAX  =  15_000  # Hz
COARSE_STEP   =     300  # Hz   -> 101 candidates
FINE_HALF     =     500  # Hz   -> +/-500 Hz window around coarse best
FINE_STEP     =      25  # Hz   -> 41 candidates


# -------------------------------------------------------------
# SRRC FILTER + PN SEQUENCE   (identical to preprocess.py)
# -------------------------------------------------------------

def _srrc_coeffs() -> np.ndarray:
    """Square-root raised cosine: span=8, SPS=8, roll-off=0.75 -> 65 taps."""
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


def _pn_seq() -> np.ndarray:
    """128-bit PN (x^7+x^6+1, init=[0,0,0,0,0,0,1]) in {0,1}."""
    state = [0, 0, 0, 0, 0, 0, 1]
    bits  = []
    for _ in range(N_PN):
        out = state[6]
        bits.append(out)
        fb  = state[6] ^ state[5]
        state = [fb] + state[:6]
    return np.array(bits, dtype=np.float32)


H            = _srrc_coeffs()               # 65-tap SRRC filter
SYNC_BIPOLAR = (2 * _pn_seq() - 1).astype(np.float32)  # PN in {-1,+1}


# -------------------------------------------------------------
# LOW-LEVEL SIGNAL PROCESSING
# -------------------------------------------------------------

def load_iq(y_path: str) -> np.ndarray:
    """Load Y_test CSV (2-col: I, Q) -> complex I+jQ array (200 k samples)."""
    rows = []
    with open(y_path) as f:
        for row in csv.reader(f):
            rows.append((float(row[0]), float(row[1])))
    a = np.array(rows)
    return (a[:, 0] + 1j * a[:, 1]).astype(np.complex64)


def to_symbols(rx_iq: np.ndarray) -> np.ndarray:
    """
    Full Receiver.m front-end in Python.
    SRRC matched filter -> normalize -> eye-diagram timing -> x8 downsample.
    Returns complex symbol array (~25 000 symbols for 200 k input samples).
    """
    xx       = np.convolve(H, rx_iq)
    nv       = max(np.max(np.abs(np.real(xx))), np.max(np.abs(np.imag(xx))))
    rx_sig   = xx / nv

    # Eye diagram: pick sampling offset that maximises signal power variance
    n_frames = len(rx_sig) // SPS
    usable   = n_frames * SPS
    pwr      = (np.abs(rx_sig[:usable]) ** 2).reshape(n_frames, SPS).T
    eye_var  = pwr.sum(axis=1) / n_frames
    offset   = int(np.argmax(eye_var[:SPS])) + 1   # +1: MATLAB 1-indexed convention

    n_syms  = (len(rx_sig) - offset) // SPS
    indices = offset + np.arange(n_syms) * SPS
    return rx_sig[indices[indices < len(rx_sig)]]


def diff_decode_bits(symbols: np.ndarray) -> np.ndarray:
    """
    DBPSK differential decoder on complex symbol array.
    Returns bipolar {-1,+1} decoded bit stream (same length as symbols).

    Decision rule: |Δφ_k| < pi/2  -> '0',  |Δφ_k| > pi/2  -> '1'.
    Wrap-correction: differences > 3pi/2 are set to 0 (happen when phase
    crosses the +/-pi boundary for a '0' bit).
    """
    angles     = np.angle(symbols)
    angle_arr  = np.concatenate([[0.0], angles])
    adiff      = np.abs(angle_arr[1:] - angle_arr[:-1])
    adiff[adiff > 3 * np.pi / 2] = 0.0
    return np.sign(adiff - np.pi / 2)


def pn_correlate(bits_bip: np.ndarray):
    """
    Slide SYNC_BIPOLAR across bits_bip; return (peak_height, peak_position).
    Peak height ∈ [0, 128]: 128 = perfect match, 64 = random.
    """
    n = len(bits_bip) - N_PN
    if n <= 0:
        return 0.0, 0
    corval   = np.array([np.dot(SYNC_BIPOLAR, bits_bip[i: i + N_PN])
                         for i in range(n)])
    idx      = int(np.argmax(np.abs(corval)))
    return float(np.abs(corval[idx])), idx


# -------------------------------------------------------------
# TWO-STAGE AFC LOOP
# -------------------------------------------------------------

def _apply_comp(symbols: np.ndarray, f_hz: float) -> np.ndarray:
    """
    Symbol-level frequency compensation.
    s_comp[k] = s[k] . exp(-j . 2pi . f . k / f_sym)
    Removes a carrier offset of f_hz Hz (at symbol rate f_sym = 200 kHz).
    """
    k   = np.arange(len(symbols), dtype=np.float64)
    rot = np.exp(-1j * 2.0 * np.pi * f_hz * k / SYMBOL_RATE)
    return (symbols * rot).astype(np.complex64)


def _squared_spectrum_score(symbols: np.ndarray, f_hz: float) -> float:
    """
    AFC objective: |mean( s_comp[k]² )| after frequency compensation.

    For BPSK, s[k] ∈ {+/-A.exp(jφ)}. Squaring removes the +/-1 modulation:
        s_comp[k]² = s[k]² . exp(-2j . 2pi . f_est . k / f_sym)

    When f_est = f_true, all s_comp[k]² point in the same complex direction
    (the residual carrier phase squared), so |mean(s_comp[k]²)| is maximised.

    Derivation: follows the standard BPSK ML frequency estimator
    (Kay, "Fundamentals of Statistical Signal Processing", Vol. II, 1993).
    """
    sc  = _apply_comp(symbols, f_hz)
    return float(np.abs(np.mean(sc.astype(np.complex128) ** 2)))


def coarse_afc(symbols: np.ndarray) -> float:
    """
    Coarse frequency search over [-15, +15] kHz in 300 Hz steps (101 candidates).
    Metric: |mean(s_comp²)| — maximised when frequency offset is fully corrected.

    Note: PN-peak-based AFC fails here because the 9 kHz carrier offset only
    introduces a 16°/symbol phase increment, which falls well within the +/-90°
    diff-decoding margin and does not degrade bit accuracy.  The squared-spectrum
    estimator is SNR-robust and works without any prior bit decoding.
    """
    cands  = np.arange(COARSE_F_MIN, COARSE_F_MAX + COARSE_STEP, COARSE_STEP,
                       dtype=float)
    scores = np.array([_squared_spectrum_score(symbols, f) for f in cands])
    f_best = float(cands[np.argmax(scores)])
    print(f"    [Coarse AFC] best f = {f_best:+.0f} Hz  "
          f"score = {np.max(scores):.5f}  (metric: |mean(s²)|)")
    return f_best


def fine_afc(symbols: np.ndarray, f_coarse: float) -> float:
    """
    Fine refinement: +/-500 Hz around f_coarse in 25 Hz steps (41 candidates).
    Same squared-spectrum metric.  Then verify PN peak with the refined estimate.
    """
    cands  = np.arange(f_coarse - FINE_HALF,
                       f_coarse + FINE_HALF + FINE_STEP,
                       FINE_STEP, dtype=float)
    scores = np.array([_squared_spectrum_score(symbols, f) for f in cands])
    f_best = float(cands[np.argmax(scores)])
    # Diagnostic: check PN peak at fine AFC estimate
    syms_c      = _apply_comp(symbols, f_best)
    peak_afc, _ = pn_correlate(diff_decode_bits(syms_c))
    print(f"    [Fine  AFC] best f = {f_best:+.0f} Hz  "
          f"score = {np.max(scores):.5f}  PN peak = {peak_afc:.1f}/{N_PN}")
    return f_best


# -------------------------------------------------------------
# TX REFERENCE FROM X_TEST
# -------------------------------------------------------------

def clean_symbols_from_x(x_path: str) -> np.ndarray:
    """
    Load X_Test CSV (1664 samples, real-valued, DATA ONLY — no PN/header),
    apply SRRC matched filter, remove 4-symbol ramp on each end.
    Returns 204 clean bipolar {-1,+1} diff-encoded symbols.

    Matches preprocess.clean_symbols_from_x exactly.
    """
    with open(x_path) as f:
        x = np.array([float(row[0]) for row in csv.reader(f)])

    flt  = np.convolve(H, x.astype(complex))
    nv   = max(np.max(np.abs(np.real(flt))), np.max(np.abs(np.imag(flt))))
    flt  = flt / nv

    delay = (len(H) - 1) // 2            # 32 samples = 4 symbols at SPS=8
    syms  = np.real(flt[delay::SPS])
    syms  = np.sign(syms)
    ramp  = 4
    return syms[ramp: -ramp].astype(np.float32)   # 204 symbols


# -------------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------------

def _mk_scaler(ckpt) -> StandardScaler:
    sc        = StandardScaler()
    sc.mean_  = ckpt["scaler_mean"]
    sc.scale_ = ckpt["scaler_scale"]
    return sc


def load_mlp_iq():
    """MLP on raw I,Q window features (input dim = 2.WINDOW = 22)."""
    ckpt  = torch.load(os.path.join(MODEL_DIR, "mlp_iq.pt"), map_location="cpu",
                       weights_only=False)
    cfg   = ckpt["best_cfg"]
    model = MLP(2 * WINDOW, tuple(cfg["hidden_dims"]), cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, _mk_scaler(ckpt)


def load_mlp_phase():
    """MLP on differential-phase window features (input dim = WINDOW = 11)."""
    ckpt  = torch.load(os.path.join(MODEL_DIR, "mlp_phase.pt"), map_location="cpu",
                       weights_only=False)
    cfg   = ckpt["best_cfg"]
    model = MLP(WINDOW, tuple(cfg["hidden_dims"]), cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, _mk_scaler(ckpt)


def load_mlp_combined():
    """MLP on combined features: I,Q + mag + Δφ (input dim = 4.WINDOW-1 = 43)."""
    ckpt  = torch.load(os.path.join(MODEL_DIR, "mlp_combined.pt"), map_location="cpu",
                       weights_only=False)
    cfg   = ckpt["best_cfg"]
    model = MLP(4 * WINDOW - 1, tuple(cfg["hidden_dims"]), cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, _mk_scaler(ckpt)


def load_resnet():
    """ResNet equalizer on combined features (input dim = 4.WINDOW-1 = 43)."""
    ckpt  = torch.load(os.path.join(MODEL_DIR, "resnet_combined.pt"), map_location="cpu",
                       weights_only=False)
    cfg   = ckpt["best_cfg"]
    model = ResNetMLP(4 * WINDOW - 1,
                      hidden=cfg["hidden"], n_blocks=cfg["n_blocks"],
                      dropout=cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, _mk_scaler(ckpt)


def load_cvnn():
    """Complex-valued NN on I+jQ window (input dim = WINDOW = 11 per branch)."""
    ckpt  = torch.load(os.path.join(MODEL_DIR, "cvnn.pt"), map_location="cpu",
                       weights_only=False)
    cfg   = ckpt["best_cfg"]
    model = CVNN(in_features=WINDOW, hidden=tuple(cfg["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, _mk_scaler(ckpt)


def load_lstm():
    """LSTM sequence equalizer (2 inputs: I, Q per timestep)."""
    ckpt  = torch.load(os.path.join(MODEL_DIR, "lstm.pt"), map_location="cpu",
                       weights_only=False)
    cfg   = ckpt["best_cfg"]
    model = LSTMEqualizer(input_dim=2,
                          hidden=int(cfg["hidden"]),
                          n_layers=int(cfg["n_layers"]),
                          dropout=float(cfg["dropout"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, _mk_scaler(ckpt)


def load_linear():
    """Ridge regression pipeline (sklearn Pipeline: StandardScaler + Ridge)."""
    with open(os.path.join(MODEL_DIR, "linear_iq.pkl"), "rb") as f:
        return pickle.load(f)["model"]


# -------------------------------------------------------------
# PREDICTION HELPERS
# -------------------------------------------------------------

def predict_mlp_model(model, scaler, X: np.ndarray) -> np.ndarray:
    """Forward pass for MLP / ResNet / CVNN models. Returns {-1,+1} array."""
    X_s = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        return torch.sign(model(torch.from_numpy(X_s))).numpy()


def predict_lstm_model(model, scaler, rx_burst: np.ndarray) -> np.ndarray:
    """
    LSTM forward pass on full symbol burst (processed as a sequence).
    Input: complex symbols, shape (N,)  -> stacked as (1, N, 2) after scaling.
    Returns {-1,+1} array of length N.
    """
    feat = np.stack([np.real(rx_burst), np.imag(rx_burst)],
                    axis=1).astype(np.float32)
    feat_s = scaler.transform(feat)
    x      = torch.from_numpy(feat_s).unsqueeze(0)   # (1, N, 2)
    with torch.no_grad():
        return torch.sign(model(x)).squeeze(0).numpy()


def classical_decode_burst(rx_burst: np.ndarray) -> np.ndarray:
    """
    Classical DBPSK decoder on a single burst of complex symbols.
    Returns estimated original information bits in {-1, +1}.

    Differential decoding: Δφ[k] = |∠s[k] − ∠s[k−1]|
      Δφ < pi/2  -> '0',   Δφ > pi/2  -> '1'
    Original bit recovery from differential-encoded stream:
      orig[0] = d[0],   orig[k] = d[k−1] ⊕ d[k]  (for k > 0)
    """
    angles    = np.angle(rx_burst)
    angle_arr = np.concatenate([[0.0], angles])
    adiff     = np.abs(angle_arr[1:] - angle_arr[:-1])
    adiff[adiff > 3 * np.pi / 2] = 0.0
    d_bits    = (np.sign(adiff - np.pi / 2) + 1) / 2   # {0, 1}

    orig    = np.zeros(len(d_bits), dtype=int)
    orig[0] = int(d_bits[0])
    for k in range(1, len(d_bits)):
        orig[k] = int(d_bits[k - 1]) ^ int(d_bits[k])
    return (2 * orig - 1).astype(np.float32)


# -------------------------------------------------------------
# PER-FILE PIPELINE
# -------------------------------------------------------------

def process_one_file(msg_name: str, models: dict) -> dict:
    """
    Full AFC + alignment + ML evaluation for one hardware file pair.
    Returns  {model_name: BER}  (empty dict if alignment fails).
    """
    x_path = os.path.join(X_DIR, f"{msg_name}_tx.csv")
    y_path = os.path.join(Y_DIR, f"{msg_name}_rx.csv")
    print(f"\n{'-'*55}")
    print(f"  File: {msg_name}")
    print(f"{'-'*55}")

    # -- Step 1: RX front-end ----------------------------------
    rx_iq   = load_iq(y_path)
    symbols = to_symbols(rx_iq)
    print(f"  RX symbols: {len(symbols):,}  "
          f"(expected ~ {200_000 // SPS:,})")

    # Baseline: PN peak without AFC (diagnostic only)
    peak_raw, _ = pn_correlate(diff_decode_bits(symbols))
    print(f"  PN peak (no AFC): {peak_raw:.1f}/{N_PN}  "
          f"({'USABLE' if peak_raw > 80 else 'too low — AFC needed'})")

    # -- Step 2: Two-stage AFC ---------------------------------
    f_coarse = coarse_afc(symbols)
    f_best   = fine_afc(symbols, f_coarse)

    # -- Step 3: Apply compensation + verify PN sync -----------
    syms_c          = _apply_comp(symbols, f_best)
    bits_c          = diff_decode_bits(syms_c)
    peak_afc, s_idx = pn_correlate(bits_c)
    print(f"  PN peak (AFC):    {peak_afc:.1f}/{N_PN}  at symbol idx {s_idx}")

    if peak_afc < 80:
        print(f"  WARNING: PN peak {peak_afc:.1f} < 80 — alignment may be unreliable")

    # -- Step 4: Extract data burst ----------------------------
    d_start = s_idx + DATA_SKIP        # skip N_PN=128 + len_field=16 = 144 symbols
    d_end   = d_start + HW_DATA_SYMS   # 204 data symbols
    if d_end > len(syms_c):
        print(f"  ERROR: burst extraction out of bounds "
              f"(need {d_end}, have {len(syms_c)}) — skipping file")
        return {}

    rx_burst = syms_c[d_start: d_end].astype(np.complex64)
    print(f"  Data burst: [{d_start}, {d_end})  = {len(rx_burst)} symbols")

    # -- Step 5: TX reference ---------------------------------
    tx_diff  = clean_symbols_from_x(x_path)   # 204 diff-encoded bipolar symbols
    tx_ref   = orig_bits(tx_diff)             # 204 original info bits {-1,+1}
    assert len(tx_ref) == HW_DATA_SYMS

    # Windowed reference (for features that trim HALF_WIN on each side)
    n_win    = len(rx_burst) - 2 * HALF_WIN   # = 204 - 10 = 194
    tx_w     = tx_ref[HALF_WIN: HALF_WIN + n_win]

    # -- Step 6: Evaluate each model --------------------------
    # Helper: align tx_ref window to feature array length
    def tx_align(n_feats):
        return tx_ref[HALF_WIN: HALF_WIN + n_feats]

    bers = {}

    # Classical
    preds_cl          = classical_decode_burst(rx_burst)
    bers["Classical"] = float(np.mean(preds_cl != tx_ref))

    # Linear-IQ (sklearn pipeline, no separate scaler needed)
    X_iq              = make_features_iq(rx_burst)           # (194, 22)
    preds_lin         = np.sign(models["linear"].predict(X_iq))
    bers["Linear-IQ"] = float(np.mean(preds_lin != tx_align(len(X_iq))))

    # MLP-IQ
    m, sc          = models["mlp_iq"]
    bers["MLP-IQ"] = float(np.mean(
        predict_mlp_model(m, sc, X_iq) != tx_align(len(X_iq))))

    # MLP-Phase  (diff_phase has N-1 values -> one fewer feature row)
    X_ph              = make_features_phase(rx_burst)        # (193, 11)
    m, sc             = models["mlp_phase"]
    bers["MLP-Phase"] = float(np.mean(
        predict_mlp_model(m, sc, X_ph) != tx_align(len(X_ph))))

    # MLP-Combined
    X_co                 = make_features_combined(rx_burst)  # (193, 43)
    m, sc                = models["mlp_combined"]
    bers["MLP-Combined"] = float(np.mean(
        predict_mlp_model(m, sc, X_co) != tx_align(len(X_co))))

    # ResNet-Combined (same features as MLP-Combined)
    m, sc                   = models["resnet"]
    bers["ResNet-Combined"] = float(np.mean(
        predict_mlp_model(m, sc, X_co) != tx_align(len(X_co))))

    # CVNN (same features as MLP-IQ)
    m, sc        = models["cvnn"]
    bers["CVNN"] = float(np.mean(
        predict_mlp_model(m, sc, X_iq) != tx_align(len(X_iq))))

    # LSTM  (full burst as one sequence; no windowing, uses all 204 symbols)
    m, sc        = models["lstm"]
    preds_lstm   = predict_lstm_model(m, sc, rx_burst)
    bers["LSTM"] = float(np.mean(preds_lstm != tx_ref))

    # Print per-model BER summary
    print(f"\n  {'Model':<20} {'BER':>8}")
    print(f"  {'-'*30}")
    for mname, b in bers.items():
        print(f"  {mname:<20} {b:>8.4f}")

    return bers


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    print("=" * 65)
    print("  Hardware Evaluation — Full AFC Loop")
    print(f"  Coarse search: [{COARSE_F_MIN/1e3:.0f}, {COARSE_F_MAX/1e3:.0f}] kHz "
          f"in {COARSE_STEP} Hz steps")
    print(f"  Fine   search: +/-{FINE_HALF} Hz in {FINE_STEP} Hz steps")
    print("=" * 65)

    # Discover hardware files
    hw_files = sorted(
        f[:-7] for f in os.listdir(X_DIR) if f.endswith("_tx.csv")
    )
    print(f"\nFound {len(hw_files)} hardware file(s):")
    for h in hw_files:
        print(f"  {h}")

    # Load all models once
    print("\nLoading trained models ...")
    models = {
        "linear":       load_linear(),
        "mlp_iq":       load_mlp_iq(),
        "mlp_phase":    load_mlp_phase(),
        "mlp_combined": load_mlp_combined(),
        "resnet":       load_resnet(),
        "cvnn":         load_cvnn(),
        "lstm":         load_lstm(),
    }
    print("  All models loaded.\n")

    # Evaluate each file
    all_bers = {}
    for msg in hw_files:
        all_bers[msg] = process_one_file(msg, models)

    # -- Summary table -----------------------------------------
    MODEL_NAMES = ["Classical", "Linear-IQ", "MLP-IQ", "MLP-Phase",
                   "MLP-Combined", "ResNet-Combined", "CVNN", "LSTM"]

    print("\n\n" + "=" * 85)
    print("  HARDWARE BER SUMMARY")
    print("=" * 85)
    hdr = f"  {'File':<28}" + "".join(f"{n:>10}" for n in MODEL_NAMES)
    print(hdr)
    print("  " + "-" * 83)
    for msg, bd in all_bers.items():
        short = msg[:26] if len(msg) > 26 else msg
        row = f"  {short:<26}" + "".join(
            f"{bd.get(n, float('nan')):>10.4f}" for n in MODEL_NAMES)
        print(row)

    # Mean across files
    mean_bers = {}
    for n in MODEL_NAMES:
        vals = [b[n] for b in all_bers.values() if n in b and not np.isnan(b.get(n, np.nan))]
        mean_bers[n] = float(np.mean(vals)) if vals else float("nan")
    print("  " + "-" * 83)
    mean_row = f"  {'Mean (hardware)':<26}" + "".join(
        f"{mean_bers[n]:>10.4f}" for n in MODEL_NAMES)
    print(mean_row)
    print("=" * 85)

    # -- Save hardware results ---------------------------------
    hw_csv = os.path.join(RESULT_DIR, "hardware_ber_results.csv")
    with open(hw_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["File"] + MODEL_NAMES)
        for msg, bd in all_bers.items():
            w.writerow([msg] + [f"{bd.get(n, ''):.4f}" for n in MODEL_NAMES])
        w.writerow(["Mean HW"] + [f"{mean_bers.get(n, ''):.4f}" for n in MODEL_NAMES])
    print(f"\nHardware results saved: {hw_csv}")

    # -- Save combined (simulated + hardware) -----------------
    sim_csv  = os.path.join(RESULT_DIR, "ber_results.csv")
    comb_csv = os.path.join(RESULT_DIR, "ber_results_combined.csv")

    sim_rows = []
    if os.path.exists(sim_csv):
        with open(sim_csv) as f:
            for row in csv.reader(f):
                sim_rows.append(row)

    with open(comb_csv, "w", newline="") as f:
        w = csv.writer(f)
        for row in sim_rows:
            w.writerow(row)
        # Append per-file hardware rows
        for msg, bd in all_bers.items():
            short = msg[:24] if len(msg) > 24 else msg
            w.writerow([f"HW:{short}"] +
                       [f"{bd.get(n, ''):.4f}" for n in MODEL_NAMES])
        # Append mean hardware row
        w.writerow(["HW Mean"] +
                   [f"{mean_bers.get(n, ''):.4f}" for n in MODEL_NAMES])
    print(f"Combined results saved: {comb_csv}")


if __name__ == "__main__":
    main()
