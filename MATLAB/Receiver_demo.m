%% Experiment 8
close all;
clc;

%% Samples per symbol
samplesPerSymbol = 8                                                                                                                                                                                                                                                                    ;
RX_ss_tmp = readmatrix('Acquired_data.xlsx'); % Acquired data at the NI using LabView (RFSA acquired)
RX_ss = complex(RX_ss_tmp(:,1), RX_ss_tmp(:,2)); % Received data having two column (I and Q) converting into complex

FIR_coeff75 = comm.RaisedCosineTransmitFilter('Shape', 'Square root', 'RolloffFactor', 0.75, 'FilterSpanInSymbols', 8, 'OutputSamplesPerSymbol', 8).coeffs.Numerator;
XX_signal = conv(FIR_coeff75, RX_ss);
RX_signal = XX_signal / max(max(abs(real(XX_signal))), max(abs(imag(XX_signal))));  % normalize by max amplitude (matches Python)

%~~~~~~~~~~~~~~~~~~~~~~~~~~~ Eye Diagram ~~~~~~~~~~~~~~~~~~~~~~%
eye_len = samplesPerSymbol;           % eye window length = at the receiver digitizer (sample /symbol)
eye_frame_len = floor(length(RX_signal)/eye_len); %1D-----> 2D eg. 1*1000=8*1000/8 DIMENTION
I_eye = zeros(eye_len, eye_frame_len);
Q_eye = zeros(eye_len, eye_frame_len);

for i=1:eye_frame_len
    I_eye(:, i) = real(RX_signal((i-1)*eye_len+1:i*eye_len,1)); % Inphase component 1D-----> 2D eg. 1*1000=8*1000/8 DIMENTION
    Q_eye(:, i) = imag(RX_signal((i-1)*eye_len+1:i*eye_len,1)); % Q phase component 1D-----> 2D eg. 1*1000=8*1000/8 DIMENTION 
end

figure;
plot(I_eye(:,1:100));
hold on;
title("Eye Diagram of the received signal");
ylabel("Envelope");
xlabel("One eye width (8 samples)");
hold off;
%~~~~~~~~~~Sampling Instant Correction~~~~~~~~~~~~~~~~~~~~~~%
eye_var = zeros(eye_len, 1);
for i=1:eye_frame_len
    for j=1:eye_len
            eye_var(j,1) = eye_var(j, 1) + I_eye(j,i)^2 + Q_eye(j,i)^2;  % use power (I²+Q²) for phase-independent timing
    end
end

eye_var = eye_var/eye_frame_len;  % normalization because of sumation

[~, eye_offset] = max(eye_var(1:samplesPerSymbol));  % finding the high sample value

%% Sample the symbols by downsampling
Symbols = zeros(floor(length(RX_signal)/samplesPerSymbol),1);

for i=1:length(Symbols)-1
    Symbols(i,1) = RX_signal((i-1)*samplesPerSymbol+eye_offset+1,1); % +1 matches Python's 0-indexed argmax+1 convention
end

fprintf('DEBUG: first 5 symbol phases (deg): %.1f  %.1f  %.1f  %.1f  %.1f\n', ...
    angle(Symbols(1))*180/pi, angle(Symbols(2))*180/pi, ...
    angle(Symbols(3))*180/pi, angle(Symbols(4))*180/pi, angle(Symbols(5))*180/pi);

%~~~~~~~~~Diferential Decoding~~~~~~~~~~~~~~~~~%
angle_array = [0;atan2(imag(Symbols),real(Symbols))];
angle_diff = abs(angle_array(2:end)-angle_array(1:end-1));
for i=1:length(angle_diff)
    if angle_diff(i)>3*pi/2
        angle_diff(i)=0;
    end 
end

bits_rec_bipolar = sign(angle_diff-pi/2);

N = 2^7;
% h_pn = commsrc.pn('GenPoly', [7 6 0], 'InitialStates', [0 0 0 0 0 0 1], 'NumBitsOut', N);
h_pn = comm.PNSequence('Polynomial', [7 6 0], 'InitialConditions', [0 0 0 0 0 0 1], 'VariableSizeOutput', true, 'MaximumOutputSize', [N, 1]);
syncronization_bits = h_pn(N);
sync_bipolar = 2*syncronization_bits - 1;  % convert {0,1} -> {-1,+1} to match Python

fprintf('DEBUG: eye_offset = %d / 8\n', eye_offset);

%~~~~~~~~~Finding the starting bit and detect the bits (Works in good SNR)~~~~~~~~~~~~~~~~~~~~~~~%
corval = zeros(length(bits_rec_bipolar),1);
for i=1:(length(bits_rec_bipolar) - 128)
    corval(i,1)=sum(sync_bipolar.*bits_rec_bipolar(i:i+127));  % bipolar correlation (max=128, matches Python)
end

figure;
plot(corval);
hold on;
title("Correlation of the synchronization bits ");
ylabel("Correlation factor");
xlabel("Multiple packets transmitted");
hold off;

% start_bit_ind=7;
[peak, start_bit_ind] = max(abs(corval)); %maximum correlation will be beging of the fist bits
% start_bit_ind = start_bit_ind + 16 + 128;
start_bit_ind
peak
ss = sign(corval(start_bit_ind)); %% required to correct the polarity of PLL output
bits_rec = (ss*bits_rec_bipolar+1)/2;
len = bits_rec(start_bit_ind + 128 : start_bit_ind + 144);
len = string(len);
sLen = "";
for i=1:length(len)-1
    sLen = sLen + len(i);
end
sLen
len = bin2dec(sLen);

% Sanity-check decoded length (must be a multiple of 8, at most 2048 bits)
if len < 8 || len > 2048 || mod(len, 8) ~= 0
    warning('Decoded length field = %d is invalid (low SNR). Classical text unavailable.', len);
    text = '[decode failed]';
else
    detected_bits = bits_rec(start_bit_ind:start_bit_ind + 143 + len);
    bin_to_text = detected_bits(145:end);
    btxt = reshape(bin_to_text, [8, length(bin_to_text)/8])';
    if length(class(btxt)) == 6
        text = char(bin2dec(char(btxt+48)))';
    else
        text = char(bin2dec(btxt))';
    end
end
text

%% ── ML DEMO EXPORT (added for demo_live.py) ──────────────────
% Save the full burst from PN start so the Python windowed models
% receive the same context they were trained on.
% Frame layout: PN(128) + len_field(16) + data(208) = 352 symbols.
% We save 360 symbols to give HALF_WIN=5 samples of leading context.
burst_start = start_bit_ind;
n_save      = min(360, length(Symbols) - burst_start + 1);

if n_save < 200
    warning('Not enough symbols after burst start -- check SNR / alignment.');
else
    sym_slice = Symbols(burst_start : burst_start + n_save - 1);
    writematrix([real(sym_slice), imag(sym_slice)], 'demo_symbols.csv');

    % Save metadata for the Python script
    fid = fopen('demo_meta.txt', 'w');
    fprintf(fid, 'peak=%d\n',        round(peak));
    fprintf(fid, 'start=%d\n',       start_bit_ind);
    fprintf(fid, 'classical=%s\n',   text);
    fclose(fid);

    fprintf('\n--- Demo export done ---\n');
    fprintf('PN peak      : %d / 128\n', round(peak));
    fprintf('Classical out: %s\n',       text);
    fprintf('Now run      : python demo_live.py\n');
end