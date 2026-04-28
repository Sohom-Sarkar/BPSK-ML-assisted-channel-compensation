%% Experiment:8 Transmitter
clc;
close all;

%% Number of bits
N=2^7;

%% Textto binary0
fid = fopen('testingCopy.txt');
x = fread(fid,'*char');
%% lines = strsplit(x, '\n');
%% disp(lines);     
binary = dec2bin(x, 8);
binary_t = transpose(binary);
txt_to_bin = binary_t(:) - '0';

%% Adding syncronization bits
h_pn = comm.PNSequence('Polynomial', [7 6 0], 'InitialConditions', [0 0 0 0 0 0 1], 'VariableSizeOutput', true, 'MaximumOutputSize', [N, 1]);
% h_pn = commsrc.pn('GenPoly', [7 6 0], 'InitialStates', [0 0 0 0 0 0 1], 'NumBitsOut', N);
syncronization_bits = h_pn(N);
messageLength = dec2bin(length(txt_to_bin), 16);
messageLengthTransposed = transpose(messageLength);
text_to_bin_messageLength = messageLengthTransposed(:) - '0';

%% [sync bits, message length, data bits]
tx_text = [syncronization_bits; text_to_bin_messageLength; txt_to_bin];

%% Differential Encoding
diff_bits = zeros(length(tx_text), 1);
diff_bits(1) = xor(0, tx_text(1));
for i=2:length(diff_bits)
    diff_bits(i) = xor(diff_bits(i-1), tx_text(i)); % XOR ing privious bit to current bit
end

%% Bit Sequence to Signal (Upsampling)
bits_I = 2*(diff_bits - 0.5);
sig_ipt_I = upsample(bits_I, 8);
FIR_coeff75 = comm.RaisedCosineTransmitFilter('Shape', 'Square root', 'RolloffFactor', 0.75, 'FilterSpanInSymbols', 8, 'OutputSamplesPerSymbol', 8).coeffs.Numerator;
sig_RRC75_I = conv(FIR_coeff75,sig_ipt_I);
sig_RRC75_I = sig_RRC75_I/max(sig_RRC75_I);
csvwrite('Non_Coherent_Cosine_075_text_ref.csv', tx_text);
csvwrite('Non_Coherent_Cosine_075_text_data123.csv', sig_RRC75_I);



format long


Burst_frequency=(200*10^3*8)/length(sig_RRC75_I)
Trigger_interval=1/Burst_frequency
 
