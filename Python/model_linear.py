#!/usr/bin/env python3
"""
model_linear.py

Linear-IQ equalizer: Ridge regression on raw I,Q windowed features.

Hyperparameter tuning:
    - alpha (regularisation strength): GridSearchCV, 5-fold CV on training set
    - Search space: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
"""

import numpy as np
import os
import pickle

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from common import (
    load_symbol_files, build_dataset, make_features_iq,
    eval_ber, CONDITIONS, MODEL_DIR
)

# ─────────────────────────────────────────────────────────────
# HYPERPARAMETER GRID
# ─────────────────────────────────────────────────────────────
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
CV_FOLDS   = 5


def run():
    """Search Ridge alpha, train best model, return test BER dict."""
    print("\n" + "=" * 60)
    print("Linear-IQ  (Ridge Regression)")
    print("=" * 60)

    pairs    = load_symbol_files()
    pairs_tr = {l: v for l, v in pairs.items() if not l.endswith("_msg5")}
    pairs_te = {l: v for l, v in pairs.items() if l.endswith("_msg5")}

    X_tr, y_tr, _ = build_dataset(pairs_tr, make_features_iq)
    X_te, y_te, l_te = build_dataset(pairs_te, make_features_iq)
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

    # ── Hyperparameter search via GridSearchCV ────────────────
    print(f"\n  Grid search over alpha = {ALPHA_GRID}  ({CV_FOLDS}-fold CV)...")
    pipe = Pipeline([("sc", StandardScaler()), ("r", Ridge())])
    grid = GridSearchCV(
        pipe,
        param_grid={"r__alpha": ALPHA_GRID},
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_tr, y_tr)

    best_alpha = grid.best_params_["r__alpha"]
    best_cv_mse = -grid.best_score_
    print(f"\n  Best alpha  : {best_alpha}")
    print(f"  Best CV MSE : {best_cv_mse:.6f}")
    print("\n  Full CV results:")
    for mean, params in zip(-grid.cv_results_["mean_test_score"],
                             grid.cv_results_["params"]):
        marker = " <-- best" if params["r__alpha"] == best_alpha else ""
        print(f"    alpha={params['r__alpha']:<10}  CV MSE={mean:.6f}{marker}")

    # ── Final model with best alpha ───────────────────────────
    best_model = grid.best_estimator_
    preds      = np.sign(best_model.predict(X_te))
    bers       = eval_ber(preds, y_te, l_te)

    print(f"\n  Overall test BER : {np.mean(preds != y_te):.4f}")
    print(f"\n  {'Condition':<22} {'BER':>8}")
    print(f"  {'-'*32}")
    for cname in CONDITIONS:
        if cname in bers:
            print(f"  {cname:<22} {bers[cname]:>8.4f}")

    # ── Save ──────────────────────────────────────────────────
    save_path = os.path.join(MODEL_DIR, "linear_iq.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"model": best_model, "best_alpha": best_alpha}, f)
    print(f"\n  Model saved: {save_path}")

    return bers


if __name__ == "__main__":
    run()
