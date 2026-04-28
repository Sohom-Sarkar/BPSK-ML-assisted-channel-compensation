#!/usr/bin/env python3
"""
train.py

Trains and evaluates ML equalizers on the windowed BPSK symbol dataset.

Models:
    1.  Classical differential decoder         (reference, no training)
    2.  Linear-IQ                              (linear baseline on raw I,Q)
    3.  MLP-IQ                                 (raw I,Q window)
    4.  MLP-Phase                              (rotation-invariant Δφ window)
    5.  MLP-Combined                           (I,Q + |s| + Δφ)
    6.  ResNet-Combined                        (skip connections on combined features;
                                                mirrors PA additive distortion structure.
                                                Pihlajasalo et al., IEEE TWC 2023)
    7.  CVNN-MLP                               (complex-valued MLP; phase geometry
                                                preserved by processing I+jQ jointly.
                                                PMC 2025, Neural Processing Letters)
    8.  LSTM                                   (recurrent; tracks time-varying phase drift)

Training improvements vs. baseline:
    - Phase augmentation: random phase rotation on each training batch so models
      generalise across unseen phase offsets without seeing every angle explicitly.
      (DeepRx, multiple 2022-2024; DeepRx paper: random channel realisation training)
    - Cosine LR schedule + AdamW weight decay on all MLP/ResNet models.
    - BatchNorm + Dropout on all MLP layers.
    - Gradient clipping for LSTM stability.

Outputs:
    Results/ber_results.csv    -- BER table (condition x model)
    Results/ber_plots.png      -- BER bar chart per condition
    Results/models/            -- saved model files (.pt / .pkl)
"""

import numpy as np
import os, csv, pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
for d in [RESULT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

WINDOW      = 11
HALF_WIN    = WINDOW // 2
VAL_FRAC    = 0.15
RANDOM_SEED = 42

MLP_EPOCHS   = 80
MLP_LR       = 1e-3
MLP_BATCH    = 256

LSTM_HIDDEN  = 64
LSTM_LAYERS  = 2
LSTM_EPOCHS  = 80

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

CONDITIONS = {
    "SNR 100%":        lambda l: l.startswith("snr_100"),
    "SNR 75%":         lambda l: l.startswith("snr_075"),
    "SNR 50%":         lambda l: l.startswith("snr_050"),
    "SNR 25%":         lambda l: l.startswith("snr_025"),
    "Phase 0deg":      lambda l: l.startswith("phase_000"),
    "Phase 30deg":     lambda l: l.startswith("phase_030"),
    "Phase 60deg":     lambda l: l.startswith("phase_060"),
    "Phase 90deg":     lambda l: l.startswith("phase_090"),
    "Phase 120deg":    lambda l: l.startswith("phase_120"),
    "Phase 150deg":    lambda l: l.startswith("phase_150"),
    "Freq 0Hz":        lambda l: l.startswith("freq_00000"),
    "Freq 500Hz":      lambda l: l.startswith("freq_00500"),
    "Freq 1kHz":       lambda l: l.startswith("freq_01000"),
    "Freq 2kHz":       lambda l: l.startswith("freq_02000"),
    "Freq 5kHz":       lambda l: l.startswith("freq_05000"),
    "IQ alpha=1.0":    lambda l: l.startswith("iq_100"),
    "IQ alpha=0.85":   lambda l: l.startswith("iq_085"),
    "IQ alpha=0.70":   lambda l: l.startswith("iq_070"),
    "Saleh mild":      lambda l: l.startswith("saleh_mild"),
    "Saleh moderate":  lambda l: l.startswith("saleh_moderate"),
    "Saleh severe":    lambda l: l.startswith("saleh_severe"),
    "Combined":        lambda l: l.startswith("combined"),
}


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def diff_phase(syms: np.ndarray) -> np.ndarray:
    """
    Δφ_k = angle(s_k * conj(s_{k-1})) / π  ∈ [-1, 1]
    Rotation-invariant: unaffected by constant or slowly varying carrier phase.
    Length = len(syms) - 1.
    """
    return (np.angle(syms[1:] * np.conj(syms[:-1])) / np.pi).astype(np.float32)


def make_features_iq(rx: np.ndarray) -> np.ndarray:
    """Raw I,Q window — shape (N, 2*WINDOW)."""
    n = len(rx)
    out = []
    for k in range(HALF_WIN, n - HALF_WIN):
        w = rx[k - HALF_WIN: k + HALF_WIN + 1]
        out.append(np.concatenate([np.real(w), np.imag(w)]))
    return np.array(out, dtype=np.float32)


def make_features_phase(rx: np.ndarray) -> np.ndarray:
    """Differential phase window — shape (N, WINDOW). Rotation-invariant."""
    dp  = diff_phase(rx)
    n   = len(dp)
    out = []
    for k in range(HALF_WIN, n - HALF_WIN):
        out.append(dp[k - HALF_WIN: k + HALF_WIN + 1])
    return np.array(out, dtype=np.float32)


def make_features_combined(rx: np.ndarray) -> np.ndarray:
    """
    Combined: [I_window, Q_window, |s|_window, Δφ_window]
    Shape: (N, 4*WINDOW - 1).
    Ref: O'Shea & Hoydis, IEEE Trans. Cogn. Commun. Netw. 2017.
    """
    n   = len(rx)
    dp  = diff_phase(rx)
    mag = np.abs(rx).astype(np.float32)
    out = []
    for k in range(HALF_WIN, n - HALF_WIN):
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
# Randomly rotate a batch by a uniform phase to produce
# phase-invariant training signal.
# Ref: DeepRx training with random channel realisations (IEEE TWC 2021).
# ─────────────────────────────────────────────────────────────

def augment_phase(X: torch.Tensor) -> torch.Tensor:
    """
    Apply a random phase rotation to the I,Q portion of the feature vector.
    Works for both raw I,Q features (2*WINDOW cols) and combined features
    (4*WINDOW-1 cols: first 2*WINDOW are I then Q, rest are |s| and Δφ).
    The magnitude and differential-phase columns are unaffected by phase rotation
    and are left unchanged.
    Ref: DeepRx random-channel training (IEEE TWC 2021).
    """
    batch = X.shape[0]
    phi   = torch.zeros(batch, 1).uniform_(0, 2 * np.pi)
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)
    I     = X[:, :WINDOW]
    Q     = X[:, WINDOW:2 * WINDOW]
    I_rot = cos_p * I - sin_p * Q
    Q_rot = sin_p * I + cos_p * Q
    rest  = X[:, 2 * WINDOW:]          # magnitude + Δφ columns: rotation-invariant
    return torch.cat([I_rot, Q_rot, rest], dim=1)


# ─────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────

def load_symbol_files():
    pairs = {}
    for f in os.listdir(PROC_DIR):
        if f.startswith("symbols_rx_"):
            lbl    = f[len("symbols_rx_"):-len(".npy")]
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
    Xs, ys, ls = [], [], []
    for lbl, (rx, tx) in sorted(pairs.items()):
        feats   = feat_fn(rx)
        target  = orig_bits(tx)
        start   = HALF_WIN
        end     = start + len(feats)
        end     = min(end, len(target))
        feats   = feats[:end - start]
        Xs.append(feats)
        ys.append(target[start:end])
        ls.extend([lbl] * len(feats))
    return (np.concatenate(Xs), np.concatenate(ys), np.array(ls))


def train_val_test_split(X, y, labels):
    test_mask  = np.array([l.endswith("_msg5") for l in labels])
    X_tv, y_tv, l_tv = X[~test_mask], y[~test_mask], labels[~test_mask]
    X_te,  y_te, l_te = X[test_mask],  y[test_mask],  labels[test_mask]
    idx   = np.random.permutation(len(X_tv))
    nval  = int(len(X_tv) * VAL_FRAC)
    return (X_tv[idx[nval:]], y_tv[idx[nval:]], l_tv[idx[nval:]],
            X_tv[idx[:nval]], y_tv[idx[:nval]], l_tv[idx[:nval]],
            X_te, y_te, l_te)


# ─────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES
# ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class ResBlock(nn.Module):
    """
    Residual block: output = ReLU(BN(W2 * ReLU(BN(W1 * x)))) + x
    The additive skip connection structurally mirrors the additive
    distortion model of a memoryless PA nonlinearity, making ResNets
    particularly effective for Saleh-model PA compensation.
    Ref: Pihlajasalo et al., "Deep Learning OFDM Receivers", IEEE TWC 2023.
    """
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNetMLP(nn.Module):
    """
    Input projection -> stack of residual blocks -> linear output.
    Ref: Pihlajasalo et al. IEEE TWC 2023; Scientific Reports 2025.
    """
    def __init__(self, in_dim, hidden=64, n_blocks=4):
        super().__init__()
        self.proj   = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.head   = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.blocks(self.proj(x))).squeeze(1)


class ComplexLinear(nn.Module):
    """
    Complex-valued linear layer: W_r + j*W_i applied to complex input.
    Implements the Wirtinger-calculus-compatible real representation:
        [Re(out)]   [W_r  -W_i] [Re(in)]
        [Im(out)] = [W_i   W_r] [Im(in)]
    Ref: PMC 2025 dual-branch CVNN; Neural Processing Letters blind equalizer.
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
    Complex-valued MLP: I+jQ window processed with complex weights.
    Activation: modReLU on magnitude, phase preserved.
    Input: (batch, 2*WINDOW) — first WINDOW: I, second WINDOW: Q.
    Output: real-valued decision score.
    Ref: Arjovsky et al. "Unitary Evolution RNNs" (complex activation);
         PMC 2025 dual-branch CVNN for modulation classification.
    """
    def __init__(self, in_features, hidden=(32, 16)):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = in_features
        for h in hidden:
            self.layers.append(ComplexLinear(prev, h))
            prev = h
        # Real-valued output: magnitude of final complex hidden state
        self.head = nn.Linear(prev * 2, 1)

    def forward(self, x):
        x_r = x[:, :WINDOW]
        x_i = x[:, WINDOW:]
        for layer in self.layers:
            x_r, x_i = layer(x_r, x_i)
            # modReLU: relu(|z| + b) * z/|z| — approximated as relu on both parts
            x_r = torch.relu(x_r)
            x_i = torch.relu(x_i)
        out = torch.cat([x_r, x_i], dim=1)
        return self.head(out).squeeze(1)


class LSTMEqualizer(nn.Module):
    """
    Processes the full symbol burst as a sequence.
    Hidden state tracks phase drift across symbols.
    Ref: Caciularu & Burshtein, IEEE JSAIT 2020;
         Gizzini et al., IEEE GLOBECOM 2022.
    """
    def __init__(self, input_dim=2, hidden=LSTM_HIDDEN, n_layers=LSTM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers,
                            batch_first=True, dropout=0.1)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────────

def _ber(preds_t: torch.Tensor, y: np.ndarray) -> float:
    return float(np.mean(torch.sign(preds_t).numpy() != y))


def train_mlp_model(model, X_tr, y_tr, X_val, y_val,
                    scaler, name="model", phase_aug=False):
    X_tr_s  = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    ds  = TensorDataset(torch.from_numpy(X_tr_s),
                        torch.from_numpy(y_tr.astype(np.float32)))
    ldr = DataLoader(ds, batch_size=MLP_BATCH, shuffle=True)

    opt     = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MLP_EPOCHS)
    loss_fn = nn.MSELoss()

    best_val, best_state = 1.0, None
    for epoch in range(MLP_EPOCHS):
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
            val_ber = _ber(model(torch.from_numpy(X_val_s)), y_val)
        if val_ber < best_val:
            best_val  = val_ber
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            print(f"    [{name}] Epoch {epoch+1:3d}/{MLP_EPOCHS}  val_BER={val_ber:.4f}")

    model.load_state_dict(best_state)
    return model


def predict_mlp_model(model, scaler, X):
    X_s = scaler.transform(X).astype(np.float32)
    model.eval()
    with torch.no_grad():
        return torch.sign(model(torch.from_numpy(X_s))).numpy()


# ─────────────────────────────────────────────────────────────
# LSTM TRAINING
# ─────────────────────────────────────────────────────────────

def build_lstm_sequences(pairs):
    seqs = []
    for lbl, (rx, tx) in sorted(pairs.items()):
        ob   = orig_bits(tx)
        feat = np.stack([np.real(rx), np.imag(rx)], axis=1).astype(np.float32)
        seqs.append((lbl, feat, ob))
    return seqs


def train_lstm(seqs_tr, seqs_val):
    model   = LSTMEqualizer()
    opt     = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, LSTM_EPOCHS)
    loss_fn = nn.MSELoss()

    all_feat = np.concatenate([f for _, f, _ in seqs_tr])
    scaler   = StandardScaler().fit(all_feat)

    best_val, best_state = 1.0, None
    for epoch in range(LSTM_EPOCHS):
        model.train()
        for i in np.random.permutation(len(seqs_tr)):
            _, feat, lab = seqs_tr[i]
            x = torch.from_numpy(scaler.transform(feat)).unsqueeze(0)
            y = torch.from_numpy(lab).unsqueeze(0)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        errs = tot = 0
        with torch.no_grad():
            for _, feat, lab in seqs_val:
                x    = torch.from_numpy(scaler.transform(feat)).unsqueeze(0)
                pred = torch.sign(model(x)).squeeze(0).numpy()
                errs += np.sum(pred != lab); tot += len(lab)
        val_ber = errs / tot
        if val_ber < best_val:
            best_val  = val_ber
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            print(f"    [LSTM] Epoch {epoch+1:3d}/{LSTM_EPOCHS}  val_BER={val_ber:.4f}")

    model.load_state_dict(best_state)
    return model, scaler


def predict_lstm(model, scaler, seqs):
    model.eval()
    preds, labs, lbls = [], [], []
    with torch.no_grad():
        for lbl, feat, lab in seqs:
            x    = torch.from_numpy(scaler.transform(feat)).unsqueeze(0)
            pred = torch.sign(model(x)).squeeze(0).numpy()
            preds.append(pred); labs.append(lab)
            lbls.extend([lbl] * len(lab))
    return np.concatenate(preds), np.concatenate(labs), np.array(lbls)


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

def eval_ber(preds, y_te, l_te):
    out = {}
    for cname, fn in CONDITIONS.items():
        mask = np.array([fn(l) for l in l_te])
        if mask.sum() > 0:
            out[cname] = float(np.mean(preds[mask] != y_te[mask]))
    return out


def classical_ber(pairs_test):
    out = {}
    for cname, fn in CONDITIONS.items():
        bers = []
        for lbl, (rx, tx) in pairs_test.items():
            if not fn(lbl): continue
            ob      = orig_bits(tx)
            angles  = np.angle(rx)
            ang_arr = np.concatenate([[0.], angles])
            adiff   = np.abs(ang_arr[1:] - ang_arr[:-1])
            adiff[adiff > 3 * np.pi / 2] = 0.
            decoded = np.sign(adiff - np.pi / 2)
            bers.append(np.mean(decoded != ob))
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
    offsets = np.linspace(-0.375 + width/2, 0.375 - width/2, len(models))
    colors  = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860","#DA8BC3","#8C8C8C"]

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


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("Loading symbol files...")
    pairs     = load_symbol_files()
    pairs_tr  = {l: v for l, v in pairs.items() if not l.endswith("_msg5")}
    pairs_te  = {l: v for l, v in pairs.items() if l.endswith("_msg5")}
    print(f"  Train: {len(pairs_tr)} files   Test: {len(pairs_te)} files")

    all_bers = {}

    # ── Classical ────────────────────────────────────────────
    print("\nClassical differential decode BER...")
    all_bers["Classical"] = classical_ber(pairs_te)

    # ── Feature datasets ──────────────────────────────────────
    feat_configs = [
        # (name,  feat_fn,              model_factory,                    phase_aug, save_key)
        ("MLP-IQ",
         make_features_iq,
         lambda d: MLP(d, (64, 32)),
         False, "mlp_iq"),

        ("MLP-Phase",
         make_features_phase,
         lambda d: MLP(d, (64, 32)),
         False, "mlp_phase"),

        ("MLP-Combined",
         make_features_combined,
         lambda d: MLP(d, (128, 64, 32)),
         False, "mlp_combined"),

        # ResNet on combined features with phase augmentation
        # Literature: Pihlajasalo et al. IEEE TWC 2023 (ResNet for PA distortion)
        ("ResNet-Combined",
         make_features_combined,
         lambda d: ResNetMLP(d, hidden=64, n_blocks=4),
         True,  "resnet_combined"),

        # CVNN on raw I,Q (only model that processes complex signal natively)
        # Literature: PMC 2025 dual-branch CVNN; Neural Processing Letters
        ("CVNN",
         make_features_iq,
         lambda d: CVNN(WINDOW, hidden=(32, 16)),
         False, "cvnn"),
    ]

    for fname, ffn, model_fn, aug, save_key in feat_configs:
        print(f"\n{'='*60}")
        print(f"Building {fname} dataset...")
        X_tr, y_tr, l_tr = build_dataset(pairs_tr, ffn)
        X_te, y_te, l_te = build_dataset(pairs_te, ffn)
        print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

        idx  = np.random.permutation(len(X_tr))
        nval = int(len(X_tr) * VAL_FRAC)
        X_v, y_v = X_tr[idx[:nval]], y_tr[idx[:nval]]
        X_t, y_t = X_tr[idx[nval:]], y_tr[idx[nval:]]

        scaler = StandardScaler()
        model  = model_fn(X_tr.shape[1])
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Training {fname}  ({n_params:,} params, phase_aug={aug})...")

        train_mlp_model(model, X_t, y_t, X_v, y_v, scaler, name=fname, phase_aug=aug)

        preds = predict_mlp_model(model, scaler, X_te)
        all_bers[fname] = eval_ber(preds, y_te, l_te)
        print(f"  Overall test BER: {np.mean(preds != y_te):.4f}")

        torch.save({"state_dict": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_},
                   os.path.join(MODEL_DIR, f"{save_key}.pt"))

    # ── LSTM ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training LSTM...")
    seqs_all = build_lstm_sequences(pairs)
    seqs_tr  = [(l, f, b) for l, f, b in seqs_all if not l.endswith("_msg5")]
    seqs_te  = [(l, f, b) for l, f, b in seqs_all if l.endswith("_msg5")]
    idx      = np.random.permutation(len(seqs_tr))
    nval     = int(len(seqs_tr) * VAL_FRAC)
    seqs_v   = [seqs_tr[i] for i in idx[:nval]]
    seqs_t   = [seqs_tr[i] for i in idx[nval:]]

    n_params = sum(p.numel() for p in LSTMEqualizer().parameters())
    print(f"  ({n_params:,} params, hidden={LSTM_HIDDEN}, layers={LSTM_LAYERS})")
    lstm_model, lstm_scaler = train_lstm(seqs_t, seqs_v)
    preds_l, labs_l, lbls_l = predict_lstm(lstm_model, lstm_scaler, seqs_te)
    all_bers["LSTM"] = eval_ber(preds_l, labs_l, lbls_l)
    print(f"  Overall test BER: {np.mean(preds_l != labs_l):.4f}")
    torch.save({"state_dict": lstm_model.state_dict(),
                "scaler_mean": lstm_scaler.mean_,
                "scaler_scale": lstm_scaler.scale_},
               os.path.join(MODEL_DIR, "lstm.pt"))

    # ── Linear baseline ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training Linear-IQ (Ridge regression)...")
    X_tr_iq, y_tr_iq, _ = build_dataset(pairs_tr, make_features_iq)
    X_te_iq, y_te_iq, l_te_iq = build_dataset(pairs_te, make_features_iq)
    lin = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=1.0))])
    lin.fit(X_tr_iq, y_tr_iq)
    lin_preds = np.sign(lin.predict(X_te_iq))
    all_bers["Linear-IQ"] = eval_ber(lin_preds, y_te_iq, l_te_iq)
    print(f"  Overall test BER: {np.mean(lin_preds != y_te_iq):.4f}")
    with open(os.path.join(MODEL_DIR, "linear_iq.pkl"), "wb") as f:
        pickle.dump(lin, f)

    # ── BER Table ─────────────────────────────────────────────
    cond_names  = [c for c in CONDITIONS if c in all_bers["Classical"]]
    model_names = list(all_bers.keys())
    col_w = 14

    print(f"\n{'Condition':<22}" + "".join(f"{m:>{col_w}}" for m in model_names))
    print("-" * (22 + col_w * len(model_names)))
    for c in cond_names:
        row = f"{c:<22}" + "".join(
            f"{all_bers[m].get(c, float('nan')):>{col_w}.4f}" for m in model_names)
        print(row)

    csv_path = os.path.join(RESULT_DIR, "ber_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Condition"] + model_names)
        for c in cond_names:
            w.writerow([c] + [f"{all_bers[m].get(c, ''):.4f}"
                               if c in all_bers.get(m, {}) else ""
                               for m in model_names])
    print(f"\nBER table saved: {csv_path}")
    plot_ber(all_bers, cond_names, os.path.join(RESULT_DIR, "ber_plots.png"))


if __name__ == "__main__":
    main()
