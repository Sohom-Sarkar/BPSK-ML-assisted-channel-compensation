# ML-Assisted DBPSK Receiver -- Group 7

## Project Overview
This project implements and evaluates machine-learning-based equalisers for a
Differential Binary Phase Shift Keying (DBPSK) communication link built on the
National Instruments (NI) USRP software-defined radio platform.

Six ML models (MLP variants, ResNet, CVNN, LSTM) are trained to decode
received symbols in the presence of hardware non-idealities: SNR variation,
static phase offset, carrier frequency offset, IQ imbalance, and nonlinear
power-amplifier distortion.

## Folder Layout

    CSL_Project/
    |-- MATLAB/           MATLAB transmitter and receiver scripts
    |-- Python/           Python ML pipeline (training, evaluation, demo)
    |-- CapturedData/     Hardware I/Q capture pairs used for evaluation
    |-- DemoData/         Three curated I/Q files for immediate demo use
    |-- TrainingData/     Processed dataset used to train all ML models
    |-- TrainedModels/    Pre-trained model checkpoints (ready to use)
    |-- Results/          BER tables and publication-quality figures
    `-- Report/           LaTeX source of the final report

## Quick Start -- ONE COMMAND DEMO

Run any of the three curated demo files (all pre-verified):

    # A: Clean baseline -- classical + MLP-Combined both decode perfectly
    python Python/demo.py DemoData/Y_test/demo_A_clean.csv

    # B: Classical header fails, ML recovers 23-25/26 chars  <-- show this one
    python Python/demo.py DemoData/Y_test/demo_B_combined_distortion.csv

    # C: Low SNR -- both degrade, shows system limits
    python Python/demo.py DemoData/Y_test/demo_C_low_snr.csv

Or without argument (prompts for file path):
    python Python/demo.py

## Dependencies
  MATLAB  : R2021b or later, Communications Toolbox
  Python  : 3.9+  --  numpy scipy torch scikit-learn matplotlib
