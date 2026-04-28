# Demo Data -- Pre-Verified I/Q Files

Three curated received I/Q files, each confirmed to work with demo.py.
Run from the project root (CSL_Project/):

    python Python/demo.py DemoData/Y_test/demo_A_clean.csv
    python Python/demo.py DemoData/Y_test/demo_B_combined_distortion.csv
    python Python/demo.py DemoData/Y_test/demo_C_low_snr.csv

Y_test/
    demo_A_clean.csv
        Source condition : No non-ideality (clean baseline)
        PN peak          : 128 / 128  (perfect sync)
        Classical decode : Perfect
        ML result        : MLP-Combined matches reference exactly

    demo_B_combined_distortion.csv
        Source condition : 100% TX amplitude (high SNR) -- noise hits length header
        PN peak          : 128 / 128  (perfect sync)
        Classical decode : FAILS -- 2 corrupted bits in the 16-bit length field
                           cause the classical decoder to output garbage
        ML result        : 23-25 / 26 characters correct
                           KEY DEMO: classical is brittle (header dependency),
                           ML is robust (no header needed -- window-based decode)

    demo_C_low_snr.csv
        Source condition : 50% TX amplitude (reduced SNR, harder channel)
        PN peak          : ~92 / 128
        Classical decode : Fails completely
        ML result        : Partially correct (~30-40% characters)
                           Shows system degrading gracefully at low SNR

Labels/
    Matching ground-truth CSV for each file (read automatically by demo.py).
    Contains: non_ideality, param_name, param_value, message_id, message, ...

Signal format (all files)
    200,000 samples, two-column CSV (I, Q)
    ADC rate 1.6 MHz, 8 samples/symbol, symbol rate 200 kHz
    Modulation: DBPSK  |  Pulse shaping: SRRC roll-off 0.75, span 8
    Frame: PN(128) + length_field(16) + data(208) = 352 symbols
