# Python Scripts

## Shared Infrastructure

common.py
    Central module imported by every other script.
    Contains: shared constants (WINDOW=11, HALF_WIN=5, file paths), all model
    class definitions (MLP, ResNetMLP, CVNN, LSTMEqualizer), three feature
    extractors (make_features_iq / _phase / _combined), training loop helpers,
    and evaluation utilities.
    UPDATE the path constants at the top before running on a new machine.

## Pipeline Scripts (run in order to reproduce from scratch)

generate_dataset.py
    Generates a labelled I/Q dataset covering five hardware non-idealities:
    TX power (SNR), static phase offset, carrier frequency offset, IQ
    imbalance, and Saleh amplifier nonlinearity.  Outputs to Data/ (or
    wherever DATA_DIR points in common.py).  ~500 MB output.

preprocess.py
    Applies the full Receiver.m pipeline in Python (SRRC matched filter,
    eye-diagram timing recovery, x8 downsampling, PN correlation, burst
    extraction) to every file produced by generate_dataset.py, then builds
    the windowed ML feature matrix (dataset_windowed.npz).

train.py
    Trains one model (selected by --model argument) on dataset_windowed.npz.
    Saves checkpoint to Results/models/<name>.pt (or .pkl for sklearn).

run_all.py
    Calls train.py for every model in sequence.

generate_figures.py
    Reads BER CSV files and trained models; produces all PDF figures used in
    the report (BER bar charts, per-condition line plots, constellation
    diagrams, feature illustration, heatmap, pipeline block diagram,
    Saleh distortion curves).

## Evaluation

evaluate_hardware.py
    Evaluates all trained models on the captured data pairs in CapturedData/.
    Pipeline: SRRC -> normalize -> eye-diagram timing -> downsample ->
    two-stage AFC (coarse +-15 kHz / 300 Hz, fine +-500 Hz / 25 Hz) ->
    PN correlation -> burst extraction -> ML inference -> BER.
    Writes hardware_ber_results.csv.

## Demo Scripts

receiver_sim.py
    Python equivalent of Receiver_demo.m.
    Accepts any 200k-sample I/Q CSV (captured or from the training set),
    applies SRRC matched filter, runs two-stage AFC if the PN peak is low,
    finds the frame start via PN correlation, decodes classical text, and
    writes demo_symbols.csv + demo_meta.txt for demo_live.py.
    Usage:  python receiver_sim.py <path/to/rx_file.csv>
    Best input for demo:  TrainingData/Y_train/snr_100pct_msg5.csv

demo_live.py
    Loads demo_symbols.csv + demo_meta.txt and runs all six ML models,
    printing a side-by-side comparison table of classical vs ML decoded text.
    Run after Receiver_demo.m (MATLAB) or receiver_sim.py (Python).

## Individual Model Scripts

Each script trains and evaluates one model independently.

    model_mlp_iq.py       MLP-IQ            Input: raw I/Q window (22-dim)
    model_mlp_phase.py    MLP-Phase         Input: diff-phase window (11-dim)
    model_mlp_combined.py MLP-Combined      Input: I/Q + mag + phase (43-dim)
    model_resnet.py       ResNet-Combined   Input: same as MLP-Combined
    model_cvnn.py         CVNN              Input: complex I+jQ window (11 cmplx)
    model_lstm.py         LSTM Equaliser    Input: full I/Q burst sequence
    model_classical.py    Classical decoder Phase-threshold rule (no ML)
    model_linear.py       Ridge Regression  Input: I/Q window (22-dim)
