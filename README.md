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
    |-- DemoData/         Three I/Q files for immediate demo use
    |-- TrainingData/     Processed dataset used to train all ML models
    |-- TrainedModels/    Pre-trained model checkpoints (ready to use)
    |-- Results/          BER tables and publication-quality figures
    `-- Report/           LaTeX source of the final report

## Dependencies
  MATLAB  : R2021b or later, Communications Toolbox
  Python  : 3.9+  --  numpy scipy torch scikit-learn matplotlib
