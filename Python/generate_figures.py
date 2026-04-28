#!/usr/bin/env python3
"""
generate_figures.py

Generates all figures for the LaTeX report:
    Fig 1: System block diagram (matplotlib)
    Fig 2: Constellation plots per non-ideality
    Fig 3: BER vs SNR scale
    Fig 4: BER vs phase offset
    Fig 5: BER vs frequency offset
    Fig 6: BER vs IQ imbalance alpha
    Fig 7: Saleh AM-AM / AM-PM curves (mild / moderate / severe)
    Fig 8: Feature illustration (raw IQ vs differential phase)
    Fig 9: BER bar chart — all conditions, all models
    Fig 10: Best model per condition summary

Outputs to: Results/figures/
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

FIG_DIR = r"c:\Users\sohom\Downloads\CSL Proj\Results\figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── colour palette ──────────────────────────────────────────
C = {
    "Classical":       "#2c3e50",
    "Linear-IQ":       "#e74c3c",
    "MLP-IQ":          "#3498db",
    "MLP-Phase":       "#2ecc71",
    "MLP-Combined":    "#9b59b6",
    "ResNet-Combined": "#e67e22",
    "CVNN":            "#1abc9c",
    "LSTM":            "#e91e63",
}
MODELS = list(C.keys())

# ── BER data (from Results/ber_results.csv) ─────────────────
CONDITIONS = [
    "SNR 100%","SNR 75%","SNR 50%","SNR 25%",
    "Phase 0deg","Phase 30deg","Phase 60deg","Phase 90deg","Phase 120deg","Phase 150deg",
    "Freq 0Hz","Freq 500Hz","Freq 1kHz","Freq 2kHz","Freq 5kHz",
    "IQ alpha=1.0","IQ alpha=0.85","IQ alpha=0.70",
    "Saleh mild","Saleh moderate","Saleh severe",
    "Combined",
]

BER = {
    "Classical":       [0.0084,0.0730,0.1910,0.3764,0.0084,0.0225,0.0140,0.0253,0.0197,0.0084,0.0140,0.0169,0.0140,0.0112,0.0225,0.0309,0.0169,0.0112,0.0169,0.0112,0.0056,0.0281],
    "Linear-IQ":       [0.4711,0.4480,0.4971,0.4653,0.4422,0.4798,0.4971,0.4855,0.5202,0.5000,0.4653,0.4827,0.5145,0.4566,0.4798,0.4711,0.4538,0.4769,0.4653,0.4740,0.5173,0.5000],
    "MLP-IQ":          [0.0116,0.0434,0.1965,0.3410,0.0058,0.0029,0.0116,0.0231,0.0145,0.0260,0.0116,0.0058,0.0116,0.0029,0.0087,0.0173,0.0058,0.0000,0.0058,0.0000,0.0000,0.0202],
    "MLP-Phase":       [0.0087,0.0638,0.1855,0.3942,0.0087,0.0087,0.0087,0.0145,0.0116,0.0000,0.0058,0.0058,0.0058,0.0029,0.0058,0.0290,0.0029,0.0000,0.0000,0.0000,0.0000,0.0145],
    "MLP-Combined":    [0.0087,0.0462,0.2110,0.3786,0.0000,0.0058,0.0087,0.0231,0.0145,0.0058,0.0058,0.0058,0.0145,0.0029,0.0058,0.0202,0.0000,0.0000,0.0029,0.0000,0.0029,0.0145],
    "ResNet-Combined": [0.0087,0.0636,0.2139,0.3960,0.0058,0.0145,0.0029,0.0289,0.0116,0.0029,0.0058,0.0058,0.0202,0.0058,0.0058,0.0318,0.0029,0.0000,0.0029,0.0000,0.0029,0.0087],
    "CVNN":            [0.0116,0.0434,0.1936,0.3671,0.0029,0.0029,0.0087,0.0231,0.0145,0.0145,0.0087,0.0029,0.0173,0.0058,0.0087,0.0173,0.0029,0.0000,0.0029,0.0029,0.0000,0.0202],
    "LSTM":            [0.0084,0.0506,0.1966,0.3736,0.0112,0.0140,0.0028,0.0253,0.0197,0.0169,0.0112,0.0140,0.0140,0.0140,0.0225,0.0225,0.0084,0.0112,0.0112,0.0028,0.0000,0.0197],
}


# ════════════════════════════════════════════════════════════
# FIG 1 — System Pipeline Block Diagram
# ════════════════════════════════════════════════════════════
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.5); ax.axis("off")

    # TX chain
    tx_blocks = [
        (0.4, 1.0, "Text\nMessage"),
        (1.7, 1.0, "PN Sync\n+ Diff Enc."),
        (3.1, 1.0, "SRRC\nFilter (TX)"),
        (4.5, 1.0, "ME1000\nDAC / PA"),
    ]
    # Channel
    ch = (6.0, 1.0, "RF Channel\n(Hardware)")
    # RX chain
    rx_blocks = [
        (7.6, 1.0, "ME1000\nADC"),
        (8.9, 1.0, "SRRC\nFilter (RX)"),
        (10.2, 1.0, "Eye Diag.\nSync"),
        (11.5, 1.0, "PN Corr.\nBurst Extr."),
    ]
    # ML block
    ml_block = (12.7, 2.5, "ML\nEqualizer")
    # Decision
    dec = (12.7, 1.0, "Symbol\nDecision")

    def draw_box(cx, cy, text, color="#dce8f0", textcolor="black", w=0.95, h=0.7):
        rect = mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor="#555",
                                        linewidth=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=7.5, fontweight="normal", color=textcolor, zorder=4)

    def arrow(x1, y1, x2, y2, color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.2, mutation_scale=12), zorder=2)

    for cx, cy, txt in tx_blocks:
        draw_box(cx, cy, txt, color="#d5e8d4")
    for i in range(len(tx_blocks) - 1):
        arrow(tx_blocks[i][0]+0.48, tx_blocks[i][1],
              tx_blocks[i+1][0]-0.48, tx_blocks[i+1][1])

    draw_box(*ch, color="#fff2cc")
    arrow(tx_blocks[-1][0]+0.48, tx_blocks[-1][1], ch[0]-0.55, ch[1])

    for cx, cy, txt in rx_blocks:
        draw_box(cx, cy, txt, color="#dce8f0")
    arrow(ch[0]+0.55, ch[1], rx_blocks[0][0]-0.48, rx_blocks[0][1])
    for i in range(len(rx_blocks) - 1):
        arrow(rx_blocks[i][0]+0.48, rx_blocks[i][1],
              rx_blocks[i+1][0]-0.48, rx_blocks[i+1][1])

    # ML box (highlighted)
    draw_box(ml_block[0], ml_block[1], ml_block[2],
             color="#f8cecc", textcolor="#c0392b", w=1.0, h=0.7)
    # Decision box
    draw_box(dec[0], dec[1], dec[2], color="#dce8f0", w=1.0)

    # Arrow from last RX block up to ML, and ML down to Decision
    last_rx = rx_blocks[-1]
    ax.annotate("", xy=(ml_block[0], ml_block[1]-0.35),
                xytext=(last_rx[0]+0.0, last_rx[1]+0.35),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b",
                                lw=1.4, mutation_scale=12,
                                connectionstyle="arc3,rad=0.0"), zorder=2)
    arrow(ml_block[0], ml_block[1]-0.35, dec[0], dec[1]+0.35, color="#c0392b")

    # Labels
    ax.text(2.35, 2.0, "Transmitter (MATLAB)", fontsize=8.5,
            color="#27ae60", fontweight="bold", ha="center")
    ax.text(9.8, 2.0, "Receiver (Python / MATLAB)", fontsize=8.5,
            color="#2980b9", fontweight="bold", ha="center")
    ax.text(ml_block[0], 3.15, "ML Layer", fontsize=8.5,
            color="#c0392b", fontweight="bold", ha="center")
    ax.text(6.0, 1.85, "HW Channel\n+ Non-idealities", fontsize=7.5,
            color="#b7950b", ha="center")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_pipeline.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════
# FIG 2 — Constellation diagrams per non-ideality
# ════════════════════════════════════════════════════════════
def fig_constellations():
    rng  = np.random.default_rng(42)
    N    = 500
    SPS  = 8
    BASE = 0.00097
    SIGMA = 0.000675

    # ── Helpers ────────────────────────────────────────────
    def srrc_h():
        span, sps, b = 8, SPS, 0.75
        n = span * sps + 1
        idx = np.arange(n) - (n - 1) // 2
        t = idx / sps
        h = np.zeros(n)
        for i, ti in enumerate(t):
            if ti == 0.0:
                h[i] = 1.0 - b + 4.0 * b / np.pi
            elif abs(abs(ti) - 1.0/(4.0*b)) < 1e-9:
                h[i] = (b/np.sqrt(2.0))*((1+2/np.pi)*np.sin(np.pi/(4*b))+(1-2/np.pi)*np.cos(np.pi/(4*b)))
            else:
                num = np.sin(np.pi*ti*(1-b)) + 4*b*ti*np.cos(np.pi*ti*(1+b))
                den = np.pi*ti*(1-(4*b*ti)**2)
                h[i] = num/den
        return h

    H = srrc_h()

    def make_bpsk(bits):
        bpolar = 2.0*bits - 1.0
        up = np.zeros(len(bpolar)*SPS)
        up[::SPS] = bpolar
        sig = np.convolve(H, up)
        sig = sig / np.max(np.abs(sig))
        return sig.astype(complex)

    def apply_receive(sig, noise_std):
        rx = np.convolve(H, sig)
        rx = rx + rng.normal(0, noise_std, len(rx)) + 1j*rng.normal(0, noise_std, len(rx))
        delay = (len(H)-1)//2
        return rx[delay::SPS][:N]

    bits  = rng.integers(0, 2, N+50)
    clean = make_bpsk(bits)

    # Measure RMS amplitude of the matched-filter output for the clean signal
    # so we can set noise levels relative to actual signal power.
    _mf      = np.convolve(H, clean * BASE)
    _delay   = (len(H)-1)//2
    _sig_rms = np.std(_mf[_delay::SPS][:N].real)   # RMS of clean MF output

    # SNR levels (noise_std as fraction of _sig_rms):
    #   high SNR (~20 dB)  -> tight clusters, impairment shape clearly visible
    #   med  SNR (~10 dB)  -> moderate scatter, still shows impairment geometry
    #   low  SNR (~3 dB)   -> scattered mess, matches actual hardware SNR 25% case
    NS_HIGH = _sig_rms * 0.10
    NS_MED  = _sig_rms * 0.32
    NS_LOW  = _sig_rms * 0.70   # approximately hardware SNR 25%

    def saleh(sig, aa, ba, ap, bp):
        A   = np.abs(sig)
        Ao  = (aa*A)/(1+ba*A**2)
        phi = (ap*A**2)/(1+bp*A**2)
        return Ao * np.exp(1j*(np.angle(sig)+phi))

    # IQ imbalance: scale Q branch before passing
    iq_sig = clean * BASE
    iq_imb = np.real(iq_sig) + 1j * 0.7 * np.imag(iq_sig)

    # (signal, noise_std) pairs — noise chosen per-case for clarity
    cases = {
        "Clean / SNR 100%":   (clean * BASE,                                     NS_HIGH),
        "SNR 25%":            (clean * BASE * 0.25,                               NS_LOW),
        "Phase 90\u00b0":     (clean * BASE * np.exp(1j*np.pi/2),                NS_MED),
        "Freq offset 5 kHz":  (clean * BASE * np.exp(
                                1j*2*np.pi*5000*np.arange(len(clean))/1_600_000), NS_MED),
        "IQ imbalance \u03b1=0.7": (iq_imb,                                      NS_MED),
        "Saleh severe":       (saleh(clean, 2.0, 2.0, 3.0, 1.0) * BASE,          NS_MED),
    }

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    axes = axes.flatten()

    for ax, (title, (sig_c, ns)) in zip(axes, cases.items()):
        syms = apply_receive(sig_c, ns)
        ax.scatter(np.real(syms), np.imag(syms),
                   s=6, alpha=0.5, color="#2980b9", linewidths=0)
        ax.axhline(0, color="#ccc", lw=0.7, zorder=0)
        ax.axvline(0, color="#ccc", lw=0.7, zorder=0)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("In-Phase (I)", fontsize=7.5)
        ax.set_ylabel("Quadrature (Q)", fontsize=7.5)
        lim = max(np.std(np.real(syms)), np.std(np.imag(syms))) * 4
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle("Received Constellations Under Hardware Non-Idealities (DBPSK)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_constellations.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════
# FIG 3 — BER vs SNR scale
# ════════════════════════════════════════════════════════════
def fig_ber_snr():
    snr_labels  = ["100%", "75%", "50%", "25%"]
    snr_db      = [20*np.log10(s) for s in [1.0, 0.75, 0.50, 0.25]]
    idx         = [0, 1, 2, 3]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    skip = {"Linear-IQ"}
    for m in MODELS:
        if m in skip: continue
        vals = [BER[m][i] for i in idx]
        ax.semilogy(snr_db, vals, "o-", color=C[m], label=m, lw=1.8, ms=6)

    ax.set_xlabel("Relative SNR (dB re 100%)", fontsize=11)
    ax.set_ylabel("BER", fontsize=11)
    ax.set_title("BER vs SNR Scale", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_ylim(1e-3, 1.0)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_ber_snr.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════
# FIG 4 — BER vs Phase offset
# ════════════════════════════════════════════════════════════
def fig_ber_phase():
    phase_degs = [0, 30, 60, 90, 120, 150]
    idx        = [4, 5, 6, 7, 8, 9]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    skip = {"Linear-IQ"}
    for m in MODELS:
        if m in skip: continue
        vals = [BER[m][i] for i in idx]
        ax.plot(phase_degs, vals, "o-", color=C[m], label=m, lw=1.8, ms=6)

    ax.set_xlabel("Phase Offset (degrees)", fontsize=11)
    ax.set_ylabel("BER", fontsize=11)
    ax.set_title("BER vs Static Phase Offset", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, ncol=2)
    ax.set_xticks(phase_degs)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(0, 0.10)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_ber_phase.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════
# FIG 5 — BER vs Frequency offset
# ════════════════════════════════════════════════════════════
def fig_ber_freq():
    freq_hz = [0, 500, 1000, 2000, 5000]
    idx     = [10, 11, 12, 13, 14]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    skip = {"Linear-IQ"}
    for m in MODELS:
        if m in skip: continue
        vals = [BER[m][i] for i in idx]
        ax.plot(freq_hz, vals, "o-", color=C[m], label=m, lw=1.8, ms=6)

    ax.set_xlabel("Frequency Offset (Hz)", fontsize=11)
    ax.set_ylabel("BER", fontsize=11)
    ax.set_title("BER vs Frequency Offset", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(0, 0.06)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_ber_freq.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════
# FIG 6 — Saleh AM-AM and AM-PM curves
# ════════════════════════════════════════════════════════════
def fig_saleh():
    A = np.linspace(0, 1.5, 500)

    SALEH = {
        "Mild":     (2.0, 0.5, 0.5, 1.0),
        "Moderate": (2.0, 1.0, 1.5, 1.0),
        "Severe":   (2.0, 2.0, 3.0, 1.0),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = {"Mild": "#2ecc71", "Moderate": "#e67e22", "Severe": "#e74c3c"}

    for name, (aa, ba, ap, bp) in SALEH.items():
        Ao   = (aa * A) / (1 + ba * A**2)
        phi  = np.degrees((ap * A**2) / (1 + bp * A**2))
        ax1.plot(A, Ao,  color=colors[name], lw=2.0, label=name)
        ax2.plot(A, phi, color=colors[name], lw=2.0, label=name)

    ax1.plot(A, A, "k--", lw=1.2, label="Linear (ideal)")
    ax1.set_xlabel("Input Amplitude $A_{in}$", fontsize=11)
    ax1.set_ylabel("Output Amplitude $A_{out}$", fontsize=11)
    ax1.set_title("AM–AM Characteristic", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.set_xlabel("Input Amplitude $A_{in}$", fontsize=11)
    ax2.set_ylabel("Phase Shift $\\phi$ (degrees)", fontsize=11)
    ax2.set_title("AM–PM Characteristic", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Saleh Power Amplifier Model — Three Severity Levels",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_saleh.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════
# FIG 7 — BER bar chart, all conditions (excl. Linear-IQ)
# ════════════════════════════════════════════════════════════
def fig_ber_bar():
    """
    Two-panel bar chart:
      Top panel  — SNR conditions (BER up to 0.45, differences clearly visible)
      Bottom panel — all other 18 conditions (BER zoomed to 0.06, differences visible)
    Linear-IQ excluded from both panels (uniform ~0.48, uninformative).
    """
    models_plot = [m for m in MODELS if m != "Linear-IQ"]
    n      = len(models_plot)
    width  = 0.8 / n
    offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, n)

    # ── split conditions ──────────────────────────────────────
    snr_idx   = [0, 1, 2, 3]                      # SNR 100/75/50/25
    other_idx = list(range(4, len(CONDITIONS)))    # everything else (18 conds)
    snr_conds   = [CONDITIONS[i] for i in snr_idx]
    other_conds = [CONDITIONS[i] for i in other_idx]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9),
        gridspec_kw={"height_ratios": [1, 2.2]},
    )

    # ── top panel: SNR ────────────────────────────────────────
    x1 = np.arange(len(snr_conds))
    for i, m in enumerate(models_plot):
        vals = [BER[m][j] for j in snr_idx]
        bars = ax1.bar(x1 + offsets[i], vals, width, label=m,
                       color=C[m], alpha=0.88, edgecolor="none")
        # value labels on top of each bar
        for bar, v in zip(bars, vals):
            if v > 0.005:
                ax1.text(bar.get_x() + bar.get_width()/2, v + 0.004,
                         f"{v:.3f}", ha="center", va="bottom",
                         fontsize=5.5, rotation=90)

    ax1.set_xticks(x1)
    ax1.set_xticklabels(snr_conds, fontsize=10)
    ax1.set_ylabel("BER", fontsize=11)
    ax1.set_ylim(0, 0.48)
    ax1.set_title("(a)  SNR Degradation", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, ncol=4, framealpha=0.8)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)

    # ── bottom panel: all other conditions ────────────────────
    x2 = np.arange(len(other_conds))
    for i, m in enumerate(models_plot):
        vals = [BER[m][j] for j in other_idx]
        bars = ax2.bar(x2 + offsets[i], vals, width, label=m,
                       color=C[m], alpha=0.88, edgecolor="none")
        for bar, v in zip(bars, vals):
            if v > 0.002:
                ax2.text(bar.get_x() + bar.get_width()/2, v + 0.0005,
                         f"{v:.3f}", ha="center", va="bottom",
                         fontsize=4.5, rotation=90)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(other_conds, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("BER", fontsize=11)
    ax2.set_ylim(0, 0.060)
    ax2.set_title("(b)  Phase / Frequency / IQ Imbalance / Saleh PA / Combined  "
                  "(y-axis zoomed to 0.06)", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)

    # shared super-title
    fig.suptitle("BER per Channel Condition — All Models  (Linear-IQ excluded, BER\u22480.48 uniformly)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_ber_bar.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════
# FIG 8 — Best model per condition heatmap
# ════════════════════════════════════════════════════════════
def fig_heatmap():
    models_plot = [m for m in MODELS if m != "Linear-IQ"]
    data = np.array([BER[m] for m in models_plot])   # (n_models, n_cond)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=0.15, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("BER", fontsize=10)

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(models_plot)))
    ax.set_yticklabels(models_plot, fontsize=9)
    ax.set_title("BER Heatmap — Green = Low Error, Red = High Error",
                 fontsize=11, fontweight="bold")

    # Annotate cells
    for i, m in enumerate(models_plot):
        for j in range(len(CONDITIONS)):
            v = BER[m][j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=5.5,
                    color="white" if v > 0.08 else "black")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_heatmap.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════
# FIG 9 — Feature illustration: raw IQ vs differential phase
# ════════════════════════════════════════════════════════════
def fig_features():
    rng   = np.random.default_rng(7)
    N     = 40
    bits  = rng.integers(0, 2, N)
    # Simulate differential encode and some phase rotation
    diff  = np.zeros(N, dtype=int)
    diff[0] = bits[0]
    for i in range(1, N):
        diff[i] = diff[i-1] ^ bits[i]
    bpolar = 2.0*diff - 1.0
    phi0   = np.pi / 3   # 60 degree offset
    syms   = bpolar.astype(complex) * np.exp(1j * phi0)
    syms  += rng.normal(0, 0.12, N) + 1j*rng.normal(0, 0.12, N)

    dp = np.angle(syms[1:] * np.conj(syms[:-1])) / np.pi

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=False)

    t = np.arange(N)
    axes[0].step(t, np.real(syms), where="mid", color="#2980b9", lw=1.5)
    axes[0].step(t, np.imag(syms), where="mid", color="#e74c3c", lw=1.5, linestyle="--")
    axes[0].set_ylabel("Amplitude", fontsize=10)
    axes[0].set_title("Raw I (blue) and Q (red) — 60° phase offset visible in Q having energy",
                      fontsize=9)
    axes[0].legend(["I", "Q"], fontsize=8); axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].step(np.arange(N-1), dp, where="mid", color="#9b59b6", lw=1.5)
    axes[1].axhline(0, color="#ccc", lw=0.8)
    axes[1].axhline(1.0, color="#e74c3c", lw=0.8, ls="--", label="π (transition)")
    axes[1].axhline(-1.0, color="#e74c3c", lw=0.8, ls="--")
    axes[1].set_ylabel("Δφ / π", fontsize=10)
    axes[1].set_title("Differential Phase — Rotation-Invariant (same regardless of phase offset)",
                      fontsize=9)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    axes[2].step(np.arange(N-1), np.sign(dp), where="mid", color="#27ae60", lw=1.5)
    axes[2].set_yticks([-1, 0, 1]); axes[2].set_ylabel("Decoded bit", fontsize=10)
    axes[2].set_title("Hard decision on Δφ → recovered bits", fontsize=9)
    axes[2].set_xlabel("Symbol index", fontsize=10)
    axes[2].grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Feature Engineering: Why Differential Phase is Phase-Invariant",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_features.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("Generating figures...")
    fig_pipeline()
    fig_constellations()
    fig_ber_snr()
    fig_ber_phase()
    fig_ber_freq()
    fig_saleh()
    fig_ber_bar()
    fig_heatmap()
    fig_features()
    print(f"\nAll figures saved to: {FIG_DIR}")
