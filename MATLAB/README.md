# MATLAB Scripts

Transmitter.m
    Full DBPSK transmitter chain.
    Reads plaintext from testingCopy.txt, prepends a 128-bit PN preamble and a
    16-bit length field, differentially encodes, upsamples (x8 at 1.6 MHz),
    filters through SRRC (roll-off 0.75, span 8), and writes the baseband
    waveform to CSV for NI playback.
    Also prints the burst frequency and trigger interval.

Receiver.m
    Original DBPSK receiver (classical, no ML).
    Loads a captured I/Q CSV, applies the matched SRRC filter, reconstructs
    the eye diagram, finds the optimal sampling instant, downsamples to symbol
    rate, differentially decodes, and recovers the transmitted text via
    PN-based frame synchronisation.

Receiver_demo.m
    Drop-in replacement for Receiver.m with an ML export stage appended.
    After classical decoding it writes two files for demo_live.py:
      demo_symbols.csv  -- complex baseband symbols from PN start (360 symbols)
      demo_meta.txt     -- PN correlation peak, frame start index, decoded text
    Run this first, then:  python Python/demo_live.py

Usage
    1. Change the csvread(...) path to your captured I/Q file.
    2. Run the script in MATLAB.
    3. Check the eye diagram and correlation plot (peak should be > 70/128).
    4. Proceed to demo_live.py for the ML comparison.
