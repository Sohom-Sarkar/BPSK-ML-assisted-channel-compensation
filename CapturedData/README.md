# Captured Hardware Data

Five I/Q data pairs recorded from the NI USRP hardware setup.
Each pair shares the same random prefix (e.g. TjFcUbSgAeIwOkV).

RX/  -- Received I/Q captures
    Format  : two-column CSV, column 1 = I (in-phase), column 2 = Q (quadrature)
    Length  : 200,000 samples per file
    Rate    : 1.6 MHz ADC (8 samples per symbol at 200 kHz symbol rate)
    Source  : NI RFSA captured via LabVIEW

TX/  -- Transmitter reference signals
    Format  : single-column CSV, real-valued
    Content : upsampled data payload at 8 SPS (no PN preamble, no length header,
              no SRRC filter -- raw upsampled bits only)
    Purpose : ground-truth reference for BER calculation in evaluate_hardware.py

Signal Parameters
    Symbol rate    : 200 kHz
    ADC/DAC rate   : 1.6 MHz  (8 samples per symbol)
    Modulation     : DBPSK
    Pulse shaping  : SRRC, roll-off = 0.75, span = 8 symbols  (65 taps)
    Frame layout   : PN(128) + length_field(16) + data(208) = 352 symbols total

Processing Note
    These captures contain a carrier frequency offset from hardware oscillator
    drift.  Use receiver_sim.py (Python) or Receiver_demo.m (MATLAB) -- both
    apply automatic frequency correction -- before running ML inference.
    Due to the hardware SNR in these captures, the ML comparison demo is better
    demonstrated using the files in TrainingData/Y_train/ (see that folder).
