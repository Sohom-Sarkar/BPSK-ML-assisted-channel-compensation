# Results

CSV Files
    ber_results.csv
        Per-condition, per-model BER on the training dataset (22 x 8 table).
    ber_results_combined.csv
        Aggregated BER used by generate_figures.py.
    hardware_ber_results.csv
        BER from evaluate_hardware.py on the five captured data pairs.

Figures  (PDF, IEEEtran-compatible)
    fig_ber_bar.pdf        Side-by-side bar charts: SNR panel + non-SNR panel
    fig_ber_snr.pdf        BER vs SNR level, all models
    fig_ber_phase.pdf      BER vs phase offset, all models
    fig_ber_freq.pdf       BER vs frequency offset, all models
    fig_constellations.pdf Received constellation diagrams for key conditions
    fig_features.pdf       Feature-extraction window illustration
    fig_heatmap.pdf        BER heatmap across all conditions and models
    fig_pipeline.pdf       System block diagram
    fig_saleh.pdf          Saleh AM-AM and AM-PM distortion curves

Reproduce with:  python Python/generate_figures.py
