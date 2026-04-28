#!/usr/bin/env python3
"""
model_lstm.py

LSTM equalizer processing the full symbol burst as a sequence (I, Q) at each step.
Hidden state accumulates phase drift across the burst — effective for frequency offset.

Ref: Caciularu & Burshtein, IEEE JSAIT 2020; Gizzini et al., IEEE GLOBECOM 2022.

Hyperparameter tuning:
    Random search over N_SEARCH configurations.
    Each config trained for SEARCH_EPOCHS, best retrained for FINAL_EPOCHS.

Search space:
    hidden    : [32, 64, 128]
    n_layers  : [1, 2, 3]
    lr        : [2e-4, 5e-4, 1e-3]
    dropout   : [0.0, 0.1, 0.2]
"""

import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from common import (
    load_symbol_files, orig_bits, LSTMEqualizer,
    eval_ber, CONDITIONS, RANDOM_SEED, MODEL_DIR
)

SEARCH_SPACE = {
    "hidden":   [32, 64, 128],
    "n_layers": [1, 2, 3],
    "lr":       [2e-4, 5e-4, 1e-3],
    "dropout":  [0.0, 0.1, 0.2],
}
N_SEARCH      = 6
SEARCH_EPOCHS = 30
FINAL_EPOCHS  = 60


def sample_configs(rng, n):
    return [
        {k: rng.choice(v).item() for k, v in SEARCH_SPACE.items()}
        for _ in range(n)
    ]


def build_sequences(pairs):
    """Convert symbol pairs to (label, feat, orig_bit_labels) tuples."""
    seqs = []
    for lbl, (rx, tx) in sorted(pairs.items()):
        feat = np.stack([np.real(rx), np.imag(rx)], axis=1).astype(np.float32)
        seqs.append((lbl, feat, orig_bits(tx)))
    return seqs


def train_lstm_model(cfg, seqs_tr, seqs_val, epochs, verbose=False):
    """Train one LSTM config; return (val_ber, model, scaler)."""
    model   = LSTMEqualizer(input_dim=2,
                             hidden=cfg["hidden"],
                             n_layers=cfg["n_layers"],
                             dropout=cfg["dropout"])
    opt     = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.MSELoss()

    all_feat = np.concatenate([f for _, f, _ in seqs_tr])
    scaler   = StandardScaler().fit(all_feat)

    rng_idx  = np.random.default_rng(RANDOM_SEED)
    best_val, best_state = 1.0, None

    for epoch in range(epochs):
        model.train()
        for i in rng_idx.permutation(len(seqs_tr)):
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
                errs += int(np.sum(pred != lab))
                tot  += len(lab)
        val_ber = errs / tot
        if val_ber < best_val:
            best_val   = val_ber
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    [LSTM] Epoch {epoch+1:3d}/{epochs}  val_BER={val_ber:.4f}")

    model.load_state_dict(best_state)
    return best_val, model, scaler


def predict_lstm_model(model, scaler, seqs):
    model.eval()
    preds, labs, lbls = [], [], []
    with torch.no_grad():
        for lbl, feat, lab in seqs:
            x    = torch.from_numpy(scaler.transform(feat)).unsqueeze(0)
            pred = torch.sign(model(x)).squeeze(0).numpy()
            preds.append(pred)
            labs.append(lab)
            lbls.extend([lbl] * len(lab))
    return np.concatenate(preds), np.concatenate(labs), np.array(lbls)


def run():
    print("\n" + "=" * 60)
    print("LSTM  (sequence model, tracks phase drift across burst)")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    pairs    = load_symbol_files()
    pairs_tr = {l: v for l, v in pairs.items() if not l.endswith("_msg5")}
    pairs_te = {l: v for l, v in pairs.items() if l.endswith("_msg5")}

    seqs_all = build_sequences(pairs_tr)
    seqs_te  = build_sequences(pairs_te)

    idx  = rng.permutation(len(seqs_all))
    nval = int(len(seqs_all) * 0.15)
    seqs_val = [seqs_all[i] for i in idx[:nval]]
    seqs_tr  = [seqs_all[i] for i in idx[nval:]]
    print(f"  Train seqs: {len(seqs_tr)}  Val seqs: {len(seqs_val)}  Test seqs: {len(seqs_te)}")

    configs = sample_configs(rng, N_SEARCH)
    print(f"\n  Searching {N_SEARCH} configs × {SEARCH_EPOCHS} epochs each...")
    results = []
    for i, cfg in enumerate(configs):
        model_tmp = LSTMEqualizer(input_dim=2, hidden=cfg["hidden"],
                                   n_layers=cfg["n_layers"], dropout=cfg["dropout"])
        n_params  = sum(p.numel() for p in model_tmp.parameters())
        val_ber, _, _ = train_lstm_model(cfg, seqs_tr, seqs_val,
                                          epochs=SEARCH_EPOCHS, verbose=False)
        results.append((val_ber, cfg))
        print(f"  [{i+1:2d}/{N_SEARCH}] hidden={cfg['hidden']}  "
              f"layers={cfg['n_layers']}  lr={cfg['lr']:.0e}  "
              f"drop={cfg['dropout']}  -> val_BER={val_ber:.4f}  "
              f"({n_params:,} params)")

    results.sort(key=lambda x: x[0])
    best_ber, best_cfg = results[0]
    print(f"\n  Best config: {best_cfg}  (val_BER={best_ber:.4f})")

    print(f"\n  Retraining best config for {FINAL_EPOCHS} epochs...")
    model_tmp = LSTMEqualizer(input_dim=2, hidden=best_cfg["hidden"],
                               n_layers=best_cfg["n_layers"], dropout=best_cfg["dropout"])
    n_params = sum(p.numel() for p in model_tmp.parameters())
    print(f"  Parameters: {n_params:,}")
    final_val_ber, lstm_model, lstm_scaler = train_lstm_model(
        best_cfg, seqs_tr, seqs_val, epochs=FINAL_EPOCHS, verbose=True)
    print(f"  Final val_BER: {final_val_ber:.4f}")

    preds_l, labs_l, lbls_l = predict_lstm_model(lstm_model, lstm_scaler, seqs_te)
    bers = eval_ber(preds_l, labs_l, lbls_l)

    print(f"\n  Overall test BER : {np.mean(preds_l != labs_l):.4f}")
    print(f"\n  {'Condition':<22} {'BER':>8}")
    print(f"  {'-'*32}")
    for cname in CONDITIONS:
        if cname in bers:
            print(f"  {cname:<22} {bers[cname]:>8.4f}")

    save_path = os.path.join(MODEL_DIR, "lstm.pt")
    torch.save({
        "state_dict":   lstm_model.state_dict(),
        "scaler_mean":  lstm_scaler.mean_,
        "scaler_scale": lstm_scaler.scale_,
        "best_cfg":     best_cfg,
    }, save_path)
    print(f"\n  Model saved: {save_path}")

    return bers


if __name__ == "__main__":
    run()
