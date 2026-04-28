# Training Dataset

This folder contains the dataset used to train all ML models.

## dataset_windowed.npz  (3.9 MB)
The processed windowed feature matrix -- the direct input to train.py.
Load with numpy:

    import numpy as np
    d = np.load('dataset_windowed.npz', allow_pickle=True)
    X    = d['X']     # float32, shape (N, 22)  -- I/Q window features per symbol
    y    = d['y']     # float32, shape (N,), values in {-1, +1} -- clean reference bits
    meta = d['meta']  # object array of dicts {label, sym_idx}

N = total windowed samples across all 110 training files.
Each row is one symbol's 11-symbol I/Q context window.
Each label y[i] is the clean reference bit at the centre of that window.

## dataset_index.csv  (tiny)
Master index of all 110 training files listing condition, parameter value,
message ID.  Used by preprocess.py and generate_figures.py.

## Y_train/  (selected high-SNR files for demo)
A subset of the generated received I/Q files included for immediate use
with receiver_sim.py without needing to regenerate the full dataset.

    snr_100pct_msg5.csv  --  100% amplitude, message = "abcdefghijklmnopqrstuvwxyz"
    snr_075pct_msg5.csv  --  75%  amplitude, same message
    snr_050pct_msg5.csv  --  50%  amplitude, same message
    snr_025pct_msg5.csv  --  25%  amplitude (hard case), same message

Use for demo:
    python Python/receiver_sim.py TrainingData/Y_train/snr_100pct_msg5.csv
    python Python/demo_live.py

## How this dataset was produced
generate_dataset.py created 110 baseband I/Q captures (200k samples each,
1.6 MHz, 8 SPS) covering five channel conditions:
  - TX power / SNR       (4 levels: 100%, 75%, 50%, 25% of full amplitude)
  - Static phase offset  (6 values: 0, 30, 60, 90, 120, 150 deg)
  - Carrier freq offset  (5 values: 0, 500, 1k, 2k, 5k Hz)
  - IQ imbalance         (3 Q-branch amplitude scales: 1.00, 0.85, 0.70)
  - Saleh PA distortion  (3 severities: mild, moderate, severe)
  plus a combined-impairment condition (phase + freq + IQ together).
Signal parameters and noise floor were calibrated to match the real hardware
capture (Acquired_data.xlsx) from the lab session.
preprocess.py then applied the full Receiver.m pipeline and extracted the
windowed feature matrix saved as dataset_windowed.npz.

## Note on the raw Train/ folder
The full 110-file raw dataset (~500 MB) is not included in this zip.
Run generate_dataset.py followed by preprocess.py to regenerate it.
