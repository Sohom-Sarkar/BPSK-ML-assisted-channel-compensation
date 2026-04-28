#!/usr/bin/env python3
"""
demo_live.py
============
ML-assisted DBPSK decoding — called by demo.py, or run standalone
after Receiver_demo.m has written demo_symbols.csv + demo_meta.txt.

Standalone usage (MATLAB+Python workflow):
    python demo_live.py
"""

import os
import sys
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from common import (
    MLP, ResNetMLP, CVNN, LSTMEqualizer,
    make_features_iq, make_features_phase, make_features_combined,
    WINDOW, HALF_WIN, MODEL_DIR,
)

N_CHARS          = 26
N_DATABITS       = N_CHARS * 8          # 208
DATA_OFFSET      = 144                  # PN(128) + len_field(16)
DATA_BIT_START   = DATA_OFFSET - HALF_WIN   # 139  (windowed model offset)
N_DATABITS_SLICE = N_DATABITS

IN_IQ       = WINDOW * 2
IN_PHASE    = WINDOW
IN_COMBINED = WINDOW * 2 + WINDOW + (WINDOW - 1)


# ── Helpers ───────────────────────────────────────────────────

def bits_to_text(bits, n_chars=N_CHARS):
    bits = np.asarray(bits, dtype=int).flatten()
    chars = []
    for i in range(min(n_chars, len(bits) // 8)):
        byte = bits[i * 8 : i * 8 + 8]
        try:
            c = chr(int(''.join(map(str, byte)), 2))
            chars.append(c if c.isprintable() else '?')
        except Exception:
            chars.append('?')
    return ''.join(chars)


def load_ckpt(fname):
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(fname)
    ck = torch.load(path, weights_only=False, map_location='cpu')
    scaler = StandardScaler()
    scaler.mean_  = ck['scaler_mean']
    scaler.scale_ = ck['scaler_scale']
    return ck, scaler


def slice_data_bits(preds):
    s, e = DATA_BIT_START, DATA_BIT_START + N_DATABITS_SLICE
    if len(preds) < e:
        preds = np.concatenate([preds, np.zeros(e - len(preds), dtype=int)])
    return preds[s:e]


def predict_windowed(model, scaler, X):
    model.eval()
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(Xs)).squeeze().numpy()
    return (out > 0.5).astype(int)


# ── Main logic ────────────────────────────────────────────────

def main():
    sym_file  = os.path.join(ROOT, 'demo_symbols.csv')
    meta_file = os.path.join(ROOT, 'demo_meta.txt')

    if not os.path.exists(sym_file):
        print("ERROR: demo_symbols.csv not found.")
        print("  Run receiver_sim.py (Python) or Receiver_demo.m (MATLAB) first.")
        return

    sym_data = np.loadtxt(sym_file, delimiter=',')
    rx = sym_data[:, 0] + 1j * sym_data[:, 1]

    meta = {}
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    k, v = line.split('=', 1)
                    meta[k.strip()] = v.strip()

    peak          = int(meta.get('peak', 0))
    classical_out = meta.get('classical', '???')
    reference     = meta.get('reference', None)
    n_chars       = int(meta.get('n_chars', N_CHARS))  # actual message length

    snr_ok = peak >= 70
    status = "OK" if snr_ok else "LOW SNR -- outputs may be unreliable"

    print()
    print("=" * 54)
    print("  ML-Assisted DBPSK Demo")
    print("=" * 54)
    classical_failed = classical_out.startswith('[')

    print(f"  PN correlation peak : {peak} / 128  [{status}]")
    if reference:
        print(f"  Original message    : '{reference}'")
    if classical_failed:
        print(f"  Classical decoded   : FAILED (noise corrupted the frame header)")
        print(f"  Note: ML models use window-based inference -- no header needed")
    else:
        print(f"  Classical decoded   : '{classical_out}'")
    print("-" * 54)

    results = {}

    # MLP-IQ
    try:
        ck, sc = load_ckpt('mlp_iq.pt')
        cfg = ck['best_cfg']
        model = MLP(in_dim=IN_IQ, hidden=tuple(cfg['hidden_dims']),
                    dropout=cfg.get('dropout', 0.1))
        model.load_state_dict(ck['state_dict'])
        results['MLP-IQ'] = bits_to_text(
            slice_data_bits(predict_windowed(model, sc, make_features_iq(rx))), n_chars)
    except Exception as e:
        results['MLP-IQ'] = f'[{e}]'

    # MLP-Phase
    try:
        ck, sc = load_ckpt('mlp_phase.pt')
        cfg = ck['best_cfg']
        model = MLP(in_dim=IN_PHASE, hidden=tuple(cfg['hidden_dims']),
                    dropout=cfg.get('dropout', 0.1))
        model.load_state_dict(ck['state_dict'])
        results['MLP-Phase'] = bits_to_text(
            slice_data_bits(predict_windowed(model, sc, make_features_phase(rx))), n_chars)
    except Exception as e:
        results['MLP-Phase'] = f'[{e}]'

    # MLP-Combined
    try:
        ck, sc = load_ckpt('mlp_combined.pt')
        cfg = ck['best_cfg']
        model = MLP(in_dim=IN_COMBINED, hidden=tuple(cfg['hidden_dims']),
                    dropout=cfg.get('dropout', 0.1))
        model.load_state_dict(ck['state_dict'])
        results['MLP-Combined'] = bits_to_text(
            slice_data_bits(predict_windowed(model, sc, make_features_combined(rx))), n_chars)
    except Exception as e:
        results['MLP-Combined'] = f'[{e}]'

    # ResNet-Combined
    try:
        ck, sc = load_ckpt('resnet_combined.pt')
        cfg = ck['best_cfg']
        model = ResNetMLP(in_dim=IN_COMBINED, hidden=cfg['hidden'],
                          n_blocks=cfg['n_blocks'], dropout=cfg.get('dropout', 0.1))
        model.load_state_dict(ck['state_dict'])
        results['ResNet-Combined'] = bits_to_text(
            slice_data_bits(predict_windowed(model, sc, make_features_combined(rx))), n_chars)
    except Exception as e:
        results['ResNet-Combined'] = f'[{e}]'

    # CVNN
    try:
        ck, sc = load_ckpt('cvnn.pt')
        cfg = ck['best_cfg']
        model = CVNN(in_features=WINDOW, hidden=tuple(cfg['hidden']))
        model.load_state_dict(ck['state_dict'])
        results['CVNN'] = bits_to_text(
            slice_data_bits(predict_windowed(model, sc, make_features_iq(rx))), n_chars)
    except Exception as e:
        results['CVNN'] = f'[{e}]'

    # LSTM
    try:
        ck, sc = load_ckpt('lstm.pt')
        cfg = ck['best_cfg']
        model = LSTMEqualizer(input_dim=2, hidden=cfg['hidden'],
                              n_layers=cfg['n_layers'],
                              dropout=cfg.get('dropout', 0.0))
        model.load_state_dict(ck['state_dict'])
        model.eval()
        iq  = np.stack([rx.real, rx.imag], axis=1).astype(np.float32)
        inp = torch.from_numpy(iq).unsqueeze(0)
        with torch.no_grad():
            out = model(inp).squeeze().numpy()
        lstm_preds = (out > 0.5).astype(int)
        s = DATA_OFFSET
        lstm_data = lstm_preds[s : s + N_DATABITS_SLICE] if len(lstm_preds) > s else lstm_preds
        results['LSTM'] = bits_to_text(lstm_data, n_chars)
    except Exception as e:
        results['LSTM'] = f'[{e}]'

    # Print table
    print(f"  {'Model':<20}  Decoded text")
    print(f"  {'-'*20}  {'-'*28}")
    if reference:
        print(f"  {'[Reference]':<20}  '{reference}'")
        print(f"  {'-'*20}  {'-'*28}")
    classical_display = 'FAILED (header error)' if classical_failed else classical_out
    print(f"  {'Classical':<20}  '{classical_display}'")
    for name, txt in results.items():
        match = (reference and txt == reference) or (not reference and txt == classical_out)
        marker = ' <--' if match else ''
        print(f"  {name:<20}  '{txt}'{marker}")
    print("=" * 54)

    if not snr_ok:
        print()
        print("  NOTE: PN peak below threshold (need ~70+/128).")
        print("  Decoded outputs are unreliable at this SNR.")
        print("  Try increasing TX power or reducing cable attenuation.")


if __name__ == '__main__':
    main()
