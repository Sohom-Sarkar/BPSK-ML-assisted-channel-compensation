#!/usr/bin/env python3
"""
model_classical.py

Classical differential decoder — reference baseline, no training required.

Decodes via: decoded[k] = sign(|angle(s[k]) - angle(s[k-1])| - π/2)
This mirrors Receiver.m exactly and requires no hyperparameters.
"""

import numpy as np
from common import load_symbol_files, classical_ber, CONDITIONS

# ─────────────────────────────────────────────────────────────
# NO HYPERPARAMETERS — analytical decoder
# ─────────────────────────────────────────────────────────────

def run():
    """Evaluate classical differential decoder on test set (msg5). Returns BER dict."""
    print("\n" + "=" * 60)
    print("Classical Differential Decoder (no training)")
    print("=" * 60)

    pairs     = load_symbol_files()
    pairs_te  = {l: v for l, v in pairs.items() if l.endswith("_msg5")}
    print(f"  Test files: {len(pairs_te)}")

    bers = classical_ber(pairs_te)

    print(f"\n  {'Condition':<22} {'BER':>8}")
    print(f"  {'-'*32}")
    for cname in CONDITIONS:
        if cname in bers:
            print(f"  {cname:<22} {bers[cname]:>8.4f}")

    overall = np.mean(list(bers.values()))
    print(f"\n  Overall mean BER: {overall:.4f}")
    return bers


if __name__ == "__main__":
    run()
