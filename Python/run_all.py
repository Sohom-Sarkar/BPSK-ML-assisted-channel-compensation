#!/usr/bin/env python3
"""
run_all.py

Orchestrates all equalizer models sequentially:
    1. Classical (no training)
    2. Linear-IQ (Ridge + GridSearchCV)
    3. MLP-IQ    (random HP search)
    4. MLP-Phase (random HP search)
    5. MLP-Combined (random HP search)
    6. ResNet-Combined (random HP search + phase augmentation)
    7. CVNN     (random HP search)
    8. LSTM     (random HP search)

Collects per-condition BER from each model, saves:
    Results/ber_results.csv
    Results/ber_plots.png
"""

import csv
import numpy as np
import os

from common import CONDITIONS, RESULT_DIR, plot_ber

import model_classical
import model_linear
import model_mlp_iq
import model_mlp_phase
import model_mlp_combined
import model_resnet
import model_cvnn
import model_lstm


MODEL_RUNNERS = [
    ("Classical",        model_classical.run),
    ("Linear-IQ",        model_linear.run),
    ("MLP-IQ",           model_mlp_iq.run),
    ("MLP-Phase",        model_mlp_phase.run),
    ("MLP-Combined",     model_mlp_combined.run),
    ("ResNet-Combined",  model_resnet.run),
    ("CVNN",             model_cvnn.run),
    ("LSTM",             model_lstm.run),
]


def print_ber_table(all_bers, cond_names):
    model_names = list(all_bers.keys())
    col_w = 16
    print(f"\n{'=' * (22 + col_w * len(model_names))}")
    print(f"{'Condition':<22}" + "".join(f"{m:>{col_w}}" for m in model_names))
    print(f"{'-' * (22 + col_w * len(model_names))}")
    for c in cond_names:
        row = f"{c:<22}" + "".join(
            f"{all_bers[m].get(c, float('nan')):>{col_w}.4f}"
            if c in all_bers.get(m, {}) else f"{'N/A':>{col_w}}"
            for m in model_names
        )
        print(row)
    print(f"{'=' * (22 + col_w * len(model_names))}")


def save_csv(all_bers, cond_names):
    model_names = list(all_bers.keys())
    csv_path    = os.path.join(RESULT_DIR, "ber_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Condition"] + model_names)
        for c in cond_names:
            w.writerow([c] + [
                f"{all_bers[m].get(c, ''):.4f}" if c in all_bers.get(m, {}) else ""
                for m in model_names
            ])
    print(f"\nBER table saved: {csv_path}")
    return csv_path


def main():
    all_bers = {}

    for name, runner in MODEL_RUNNERS:
        print(f"\n{'#' * 60}")
        print(f"# Running: {name}")
        print(f"{'#' * 60}")
        try:
            bers = runner()
            all_bers[name] = bers
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback; traceback.print_exc()
            all_bers[name] = {}

    # ── Summary ───────────────────────────────────────────────
    cond_names = [c for c in CONDITIONS if c in all_bers.get("Classical", {})]
    print("\n\n" + "=" * 60)
    print("FINAL BER SUMMARY")
    print("=" * 60)
    print_ber_table(all_bers, cond_names)

    # ── Best model per condition ──────────────────────────────
    print("\nBest model per condition:")
    for c in cond_names:
        scores = {m: all_bers[m].get(c, float("inf")) for m in all_bers}
        best   = min(scores, key=scores.get)
        print(f"  {c:<22}  {best}  (BER={scores[best]:.4f})")

    # ── Save ──────────────────────────────────────────────────
    save_csv(all_bers, cond_names)
    plot_ber(all_bers, cond_names, os.path.join(RESULT_DIR, "ber_plots.png"))


if __name__ == "__main__":
    main()
