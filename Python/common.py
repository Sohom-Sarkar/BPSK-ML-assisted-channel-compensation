#!/usr/bin/env python3
"""
common.py

Shared constants, feature engineering, data loading, model architectures,
training helpers, and evaluation utilities used by all model scripts.
"""

import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────
DATA_DIR   = r"c:\Users\sohom\Downloads\CSL Proj\SimulatedData"
RESULT_DIR = r"c:\Users\sohom\Downloads\CSL Proj\Results"
MODEL_DIR  = os.path.join(RESULT_DIR, "models")
PROC_DIR   = os.path.join(DATA_DIR, "Processed")
for _d in [RESULT_DIR, MODEL_DIR]:
    os.makedirs(_d, exist_ok=True)

WINDOW      = 11
HALF_WIN    = WINDOW // 2
VAL_FRAC    = 0.15
RANDOM_SEED = 42

CONDITIONS = {
    "SNR 100%":       lambda l: l.startswith("snr_100"),
    "SNR 75%":        lambda l: l.startswith("snr_075"),
    "SNR 50%":        lambda l: l.startswith("snr_050"),
    "SNR 25%":        lambda l: l.startswith("snr_025"),
    "Phase 0deg":     lambda l: l.startswith("phase_000"),
    "Phase 30deg":    lambda l: l.startswith("phase_030"),
    "Phase 60deg":    lambda l: l.startswith("phase_060"),
    "Phase 90deg":    lambda l: l.startswith("phase_090"),
    "Phase 120deg":   lambda l: l.startswith("phase_120"),
    "Phase 150deg":   lambda l: l.startswith("phase_150"),
    "Freq 0Hz":       lambda l: l.startswith("freq_00000"),
    "Freq 500Hz":     lambda l: l.startswith("freq_00500"),
    "Freq 1kHz":      lambda l: l.startswith("freq_01000"),
    "Freq 2kHz":      lambda l: l.startswith("freq_02000"),
    "Freq 5kHz":      lambda l: l.startswith("freq_05000"),
    "IQ alpha=1.0":   lambda l: l.startswith("iq_100"),
    "IQ alpha=0.85":  lambda l: l.startswith("iq_085"),
    "IQ alpha=0.70":  lambda l: l.startswith("iq_070"),
    "Saleh mild":     lambda l: l.startswith("saleh_mild"),
    "Saleh moderate": lambda l: l.startswith("saleh_moderate"),
    "Saleh severe":   lambda l: l.startswith("saleh_severe"),
    "Combined":       lambda l: l.startswith("combined"),
}


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def diff_phase(syms: np.ndarray) -> np.ndarray:
    """Δφ_k = angle(s_k · conj(s_{k-1})) / π ∈ [-1, 1]. Rotation-invariant."""
    return (np.angle(syms[1:] * np.conj(syms[:-1])) / np.pi).astype(np.float32)


def make_features_iq(rx: np.ndarray) -> np.ndarray:
    """Raw I,Q window — shape (N, 2*WINDOW)."""
    out = []
    for k in range(HALF_WIN, len(rx) - HALF_WIN):
        w = rx[k - HALF_WIN: k + HALF_WIN + 1]
        out.append(np.concatenate([np.real(w), np.imag(w)]))
    return np.array(out, dtype=np.float32)


def make_features_phase(rx: np.ndarray) -> np.ndarray:
    """Differential phase window — shape (N, WINDOW). Rotation-invariant."""
    dp  = diff_phase(rx)
    out = []
    for k in range(HALF_WIN, len(dp) - HALF_WIN):
        out.append(dp[k - HALF_WIN: k + HALF_WIN + 1])
    return np.array(out, dtype=np.float32)


def make_features_combined(rx: np.ndarray) -> np.ndarray:
    """[I_window, Q_window, |s|_window, Δφ_window] — shape (N, 4*WINDOW-1)."""
    dp  = diff_phase(rx)
    mag = np.abs(rx).astype(np.float32)
    out = []
    for k in range(HALF_WIN, len(rx) - HALF_WIN):
        w        = rx[k - HALF_WIN: k + HALF_WIN + 1]
        feat_iq  = np.concatenate([np.real(w), np.imag(w)])
        feat_mag = mag[k - HALF_WIN: k + HALF_WIN + 1]
        dps      = k - HALF_WIN
        dpe      = dps + WINDOW - 1
        feat_dp  = dp[dps:dpe] if dpe <= len(dp) else np.zeros(WINDOW - 1, dtype=np.float32)
        out.append(np.concatenate([feat_iq, feat_mag, feat_dp]))
    return np.array(out, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# PHASE AUGMENTATION
# ─────────────────────────────────────────────────────────────

def augment_phase(X: torch.Tensor) -> torch.Tensor:
    """
    Random phase rotation on I,Q portion of feature vector.
    Magnitude and Δφ columns are unchanged (rotation-invariant).
    Ref: DeepRx random-channel training (IEEE TWC 2021).
    """
    batch = X.shape[0]
    phi   = torch.zeros(batch, 1).uniform_(0, 2 * np.pi)
    cos_p, sin_p = torch.cos(phi), torch.sin(phi)
    I, Q  = X[:, :WINDOW], X[:, WINDOW:2 * WINDOW]
    rest  = X[:, 2 * WINDOW:]
    return torch.cat([cos_p * I - sin_p * Q,
                      sin_p * I + cos_p * Q,
                      rest], dim=1)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_symbol_files():
    """Load all per-file symbol .npy pairs from Processed/."""
    pairs = {}
    for f in os.listdir(PROC_DIR):
        if not f.startswith("symbols_rx_"):
            continue
        lbl     = f[len("symbols_rx_"):-len(".npy")]
        rx_path = os.path.join(PROC_DIR, f"symbols_rx_{lbl}.npy")
        tx_path = os.path.join(PROC_DIR, f"symbols_tx_{lbl}.npy")
        if os.path.exists(tx_path):
            pairs[lbl] = (np.load(rx_path), np.load(tx_path))
    return pairs


def orig_bits(tx: np.ndarray) -> np.ndarray:
    """Recover pre-differential-encoding bits from diff-encoded bipolar TX."""
    tx_01   = ((tx + 1) / 2).astype(int)
    orig    = np.zeros(len(tx_01), dtype=int)
    orig[0] = tx_01[0]
    for i in range(1, len(tx_01)):
        orig[i] = tx_01[i - 1] ^ tx_01[i]
    return (2 * orig - 1).astype(np.float32)


def build_dataset(pairs: dict, feat_fn):
    """Build (X, y, labels) arrays from symbol pairs using given feature function."""
    Xs, ys, ls = [], [], []
    for lbl, (rx, tx) in sorted(pairs.items()):
        feats  = feat_fn(rx)
        target = orig_bits(tx)
        start  = HALF_WIN
        end    = min(start + len(feats), len(target))
        feats  = feats[:end - start]
        Xs.append(feats)
        ys.append(target[start:end])
        ls.extend([lbl] * len(feats))
    return np.concatenate(Xs), np.concatenate(ys), np.array(ls)


def split_train_val(X, y, labels, val_frac=VAL_FRAC, seed=RANDOM_SEED):
    """Shuffle and split into train and validation (no test — test is always msg5)."""
    rng = np.random.default_rng(seed)
    idx  = rng.permutation(len(X))
    nval = int(len(X) * val_frac)
    return (X[idx[nval:]], y[idx[nval:]], labels[idx[nval:]],
            X[idx[:nval]],  y[idx[:nval]],  labels[idx[:nval:]])


# ─────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES
# ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), dropout=0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class ResBlock(nn.Module):
    """
    Additive skip connection mirrors PA distortion model.
    Ref: Pihlajasalo et al., IEEE TWC 2023.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNetMLP(nn.Module):
    """Input projection -> residual blocks -> linear output."""
    def __init__(self, in_dim, hidden=64, n_blocks=4, dropout=0.1):
        super().__init__()
        self.proj   = nn.Sequential(nn.Linear(in_dim, hidden),
                                    nn.BatchNorm1d(hidden), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head   = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.blocks(self.proj(x))).squeeze(1)


class ComplexLinear(nn.Module):
    """
    Wirtinger-compatible complex linear layer.
    [Re(out), Im(out)] = [[Wr, -Wi], [Wi, Wr]] @ [Re(in), Im(in)]
    Ref: PMC 2025 dual-branch CVNN.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W_r = nn.Linear(in_features, out_features, bias=True)
        self.W_i = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x_r, x_i):
        return (self.W_r(x_r) - self.W_i(x_i),
                self.W_i(x_r) + self.W_r(x_i))


class CVNN(nn.Module):
    """
    Complex-valued MLP on I+jQ window.
    Input: (batch, 2*WINDOW) — first WINDOW: I, second WINDOW: Q.
    Ref: Arjovsky et al. (complex activation); PMC 2025.
    """
    def __init__(self, in_features=WINDOW, hidden=(32, 16)):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = in_features
        for h in hidden:
            self.layers.append(ComplexLinear(prev, h))
            prev = h
        self.head = nn.Linear(prev * 2, 1)

    def forward(self, x):
        x_r, x_i = x[:, :WINDOW], x[:, WINDOW:]
        for layer in self.layers:
            x_r, x_i = layer(x_r, x_i)
            x_r = torch.relu(x_r)
            x_i = torch.relu(x_i)
        return self.head(torch.cat([x_r, x_i], dim=1)).squeeze(1)


class LSTMEqualizer(nn.Module):
    """
    Processes full symbol burst as a sequence.
    Ref: Caciularu & Burshtein, IEEE JSAIT 2020.
    """
    def __init__(self, input_dim=2, hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        lstm_drop = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden, n_layers,
                            batch_first=True, dropout=lstm_drop)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────────

def train_mlp(model, X_tr, y_tr, X_val, y_val,
              lr=1e-3, batch_size=256, epochs=80,
              phase_aug=False, name="model", verbose=True):
    """
    Train an MLP/ResNet/CVNN model with AdamW + CosineAnnealingLR.
    Returns (best_val_ber, scaler) — model is updated in-place to best state.
    """
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s  = scaler.transform(X_val).astype(np.float32)

    ds  = TensorDataset(torch.from_numpy(X_tr_s),
                        torch.from_numpy(y_tr.astype(np.float32)))
    ldr = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.MSELoss()

    best_val, best_state = 1.0, None
    X_val_t = torch.from_numpy(X_val_s)
    y_val_np = y_val.astype(np.float32)

    for epoch in range(epochs):
        model.train()
        for xb, yb in ldr:
            if phase_aug:
                xb = augment_phase(xb)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_ber = float(np.mean(
                torch.sign(model(X_val_t)).numpy() != y_val_np))
        if val_ber < best_val:
            best_val   = val_ber
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    [{name}] Epoch {epoch+1:3d}/{epochs}  val_BER={val_ber:.4f}")

    model.load_state_dict(best_state)
    return best_val, scaler


def predict_mlp(model, scaler, X):
    X_s = scaler.transform(X).astype(np.float32)
    model.eval()
    with torch.no_grad():
        return torch.sign(model(torch.from_numpy(X_s))).numpy()


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

def eval_ber(preds, y_te, l_te):
    """BER per condition dict."""
    out = {}
    for cname, fn in CONDITIONS.items():
        mask = np.array([fn(l) for l in l_te])
        if mask.sum() > 0:
            out[cname] = float(np.mean(preds[mask] != y_te[mask]))
    return out


def classical_ber(pairs_test):
    """Classical differential decoder BER per condition."""
    out = {}
    for cname, fn in CONDITIONS.items():
        bers = []
        for lbl, (rx, tx) in pairs_test.items():
            if not fn(lbl):
                continue
            ob      = orig_bits(tx)
            ang_arr = np.concatenate([[0.], np.angle(rx)])
            adiff   = np.abs(ang_arr[1:] - ang_arr[:-1])
            adiff[adiff > 3 * np.pi / 2] = 0.
            decoded = np.sign(adiff - np.pi / 2)
            bers.append(float(np.mean(decoded != ob)))
        if bers:
            out[cname] = float(np.mean(bers))
    return out


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_ber(all_bers, cond_names, save_path):
    models  = list(all_bers.keys())
    x       = np.arange(len(cond_names))
    width   = 0.75 / len(models)
    offsets = np.linspace(-0.375 + width / 2, 0.375 - width / 2, len(models))
    colors  = ["#4C72B0","#DD8452","#55A868","#C44E52",
               "#8172B2","#937860","#DA8BC3","#8C8C8C"]

    fig, ax = plt.subplots(figsize=(22, 6))
    for i, (mname, color) in enumerate(zip(models, colors)):
        bers = [all_bers[mname].get(c, np.nan) for c in cond_names]
        ax.bar(x + offsets[i], bers, width, label=mname, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("BER")
    ax.set_title("BER per Condition: Classical vs ML Equalizers")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0, 0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {save_path}")
