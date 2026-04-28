#!/usr/bin/env python3
"""
model_mlp_iq.py

MLP on raw I,Q windowed features — shape (N, 2*WINDOW=22).

Hyperparameter tuning:
    Random search over N_SEARCH configurations, each trained for SEARCH_EPOCHS.
    Best config retrained for FINAL_EPOCHS. Metric: validation BER.

Search space:
    hidden_dims : [(64,32), (128,64), (128,64,32), (256,128,64), (256,128,64,32)]
    lr          : [5e-4, 1e-3, 2e-3, 3e-3]
    dropout     : [0.05, 0.1, 0.2]
    batch_size  : [128, 256, 512]
"""

import numpy as np
import os
import torch

from common import (
    load_symbol_files, build_dataset, split_train_val,
    make_features_iq, MLP, train_mlp, predict_mlp,
    eval_ber, CONDITIONS, RANDOM_SEED, MODEL_DIR
)

# ─────────────────────────────────────────────────────────────
# SEARCH CONFIGURATION
# ─────────────────────────────────────────────────────────────
SEARCH_SPACE = {
    "hidden_dims": [(64, 32), (128, 64), (128, 64, 32),
                    (256, 128, 64), (256, 128, 64, 32)],
    "lr":          [5e-4, 1e-3, 2e-3, 3e-3],
    "dropout":     [0.05, 0.1, 0.2],
    "batch_size":  [128, 256, 512],
}
N_SEARCH      = 10     # random configs to evaluate
SEARCH_EPOCHS = 40     # epochs per candidate
FINAL_EPOCHS  = 80     # epochs for best config (final model)


def sample_configs(rng, n):
    return [
        {k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()}
        for _ in range(n)
    ]


def run():
    print("\n" + "=" * 60)
    print("MLP-IQ  (raw I,Q window)")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    pairs    = load_symbol_files()
    pairs_tr = {l: v for l, v in pairs.items() if not l.endswith("_msg5")}
    pairs_te = {l: v for l, v in pairs.items() if l.endswith("_msg5")}

    X_tr_full, y_tr_full, _ = build_dataset(pairs_tr, make_features_iq)
    X_te,      y_te,      l_te = build_dataset(pairs_te, make_features_iq)

    X_tr, y_tr, _, X_val, y_val, _ = split_train_val(X_tr_full, y_tr_full,
                                                       np.zeros(len(X_tr_full)))
    print(f"  Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_te.shape}")

    # ── Random hyperparameter search ──────────────────────────
    configs = sample_configs(rng, N_SEARCH)
    print(f"\n  Searching {N_SEARCH} configs × {SEARCH_EPOCHS} epochs each...")
    results = []
    for i, cfg in enumerate(configs):
        model    = MLP(X_tr.shape[1], tuple(cfg["hidden_dims"]), cfg["dropout"])
        n_params = sum(p.numel() for p in model.parameters())
        val_ber, _ = train_mlp(model, X_tr, y_tr, X_val, y_val,
                                lr=cfg["lr"], batch_size=cfg["batch_size"],
                                epochs=SEARCH_EPOCHS, name=f"search_{i}",
                                verbose=False)
        results.append((val_ber, cfg))
        print(f"  [{i+1:2d}/{N_SEARCH}] hidden={cfg['hidden_dims']}  "
              f"lr={cfg['lr']:.0e}  drop={cfg['dropout']}  "
              f"bs={cfg['batch_size']}  -> val_BER={val_ber:.4f}  "
              f"({n_params:,} params)")

    results.sort(key=lambda x: x[0])
    best_ber, best_cfg = results[0]
    print(f"\n  Best config: {best_cfg}  (val_BER={best_ber:.4f})")

    # ── Retrain best config for full epochs ───────────────────
    print(f"\n  Retraining best config for {FINAL_EPOCHS} epochs...")
    model    = MLP(X_tr.shape[1], tuple(best_cfg["hidden_dims"]), best_cfg["dropout"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    final_val_ber, scaler = train_mlp(
        model, X_tr, y_tr, X_val, y_val,
        lr=best_cfg["lr"], batch_size=best_cfg["batch_size"],
        epochs=FINAL_EPOCHS, name="MLP-IQ", verbose=True)
    print(f"  Final val_BER: {final_val_ber:.4f}")

    # ── Evaluate on test ──────────────────────────────────────
    preds = predict_mlp(model, scaler, X_te)
    bers  = eval_ber(preds, y_te, l_te)

    print(f"\n  Overall test BER : {np.mean(preds != y_te):.4f}")
    print(f"\n  {'Condition':<22} {'BER':>8}")
    print(f"  {'-'*32}")
    for cname in CONDITIONS:
        if cname in bers:
            print(f"  {cname:<22} {bers[cname]:>8.4f}")

    # ── Save ──────────────────────────────────────────────────
    save_path = os.path.join(MODEL_DIR, "mlp_iq.pt")
    torch.save({
        "state_dict":    model.state_dict(),
        "scaler_mean":   scaler.mean_,
        "scaler_scale":  scaler.scale_,
        "best_cfg":      best_cfg,
    }, save_path)
    print(f"\n  Model saved: {save_path}")

    return bers


if __name__ == "__main__":
    run()
