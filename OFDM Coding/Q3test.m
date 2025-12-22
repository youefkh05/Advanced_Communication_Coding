clear; clc; close all;

%% ================= PARAMETERS =================
Nfft = 256;
Eb = 1;
SNR_dB = 0:2:20;
No = Eb ./ (10.^(SNR_dB/10));
std_noise = sqrt(No/(2*Nfft));

mods = {'BPSK','QPSK','16QAM'};
useCoding = [0 1];          % no code / repetition-5
channels = {'flat','selective'};

no_bits = 1e4;              % total bits
binary_data = randi([0 1],1,no_bits);

%% Rayleigh channels
h_flat = (randn + 1j*randn)/sqrt(2);
H_freq = (randn(1,Nfft)+1j*randn(1,Nfft))/sqrt(2);

%% ================= MAIN LOOP =================
for m = 1:length(mods)

    modType = mods{m};

    for c = 1:length(useCoding)
        for ch = 1:length(channels)

            BER{m,c,ch} = OFDM_BER( ...
                binary_data, no_bits, Nfft, Eb, ...
                SNR_dB, std_noise, modType, ...
                useCoding(c), channels{ch}, ...
                h_flat, H_freq);

        end
    end

    %% -------- Plot per modulation --------
    figure;
    semilogy(SNR_dB, BER{m,1,1}, 'r-o','LineWidth',2); hold on;
    semilogy(SNR_dB, BER{m,2,1}, 'b-s','LineWidth',2);
    semilogy(SNR_dB, BER{m,1,2}, 'k-^','LineWidth',2);
    semilogy(SNR_dB, BER{m,2,2}, 'm-d','LineWidth',2);
    grid on;

    xlabel('E_b/N_0 (dB)');
    ylabel('BER');
    legend('Flat-Uncoded','Flat-Rep5','Freq-Uncoded','Freq-Rep5');
    title(['OFDM ' modType ' BER vs SNR']);
end


function BER = OFDM_BER(binary_data, no_bits, Nfft, Eb, SNR_dB, std_noise, ...
                        mod_type, useCoding, channelType, h, H)

%% Modulation parameters
switch mod_type
    case 'BPSK',  bits_ps = 1; rows=32; cols=8;
    case 'QPSK',  bits_ps = 2; rows=32; cols=16;
    case '16QAM', bits_ps = 4; rows=32; cols=32;
end

bits_per_symbol = rows*cols;

%% Coding
if useCoding
    uncoded_bits = floor(bits_per_symbol/5);
else
    uncoded_bits = bits_per_symbol;
end

num_symbols = ceil(no_bits/uncoded_bits);

BER = zeros(1,length(SNR_dB));

for snr_i = 1:length(SNR_dB)

    bit_errors = 0;
    bit_idx = 1;

    for s = 1:num_symbols

        %% Get bits
        nb = min(uncoded_bits, no_bits-bit_idx+1);
        bits = binary_data(bit_idx:bit_idx+nb-1);
        bit_idx = bit_idx + nb;

        if useCoding
            bits = repelem(bits,5);
        end
        bits = [bits zeros(1,bits_per_symbol-length(bits))];

        %% Interleaver
        inter = reshape(bits, rows, cols).';
        inter = inter(:).';

        %% Mapper
        switch mod_type
            case 'BPSK'
                X = sqrt(Eb)*(2*inter-1);
            case 'QPSK'
                b = reshape(inter,2,[]).';
                X = sqrt(Eb)*((2*b(:,1)-1)+1j*(2*b(:,2)-1));
            case '16QAM'
                b = reshape(inter,4,[]).';
                I = (2*b(:,1)-1).*(2-(2*b(:,3)));
                Q = (2*b(:,2)-1).*(2-(2*b(:,4)));
                X = sqrt(Eb/2.5)*(I+1j*Q);
        end

        %% IFFT
        x = ifft(X);

        %% Channel
        if strcmp(channelType,'flat')
            y = h*x;
        else
            y = ifft(X.*H);
        end

        %% AWGN
        y = y + std_noise(snr_i)*(randn(size(y))+1j*randn(size(y)));

        %% Receiver
        Y = fft(y);

        if strcmp(channelType,'flat')
            Yeq = Y/h;
        else
            Yeq = Y./H;
        end

        %% Demapper
        switch mod_type
            case 'BPSK'
                rx_bits = real(Yeq)>0;
            case 'QPSK'
                rx_bits = reshape([real(Yeq)>0 imag(Yeq)>0].',1,[]);
            case '16QAM'
                rx_bits = reshape([ ...
                    real(Yeq)>0 imag(Yeq)>0 ...
                    abs(real(Yeq))<2 abs(imag(Yeq))<2].',1,[]);
        end

        %% De-interleaver
        deint = reshape(rx_bits, cols, rows).';
        rx_bits = deint(:).';

        %% Decode
        if useCoding
            dec = zeros(1,nb);
            for k=1:nb
                dec(k) = sum(rx_bits((k-1)*5+1:k*5))>2.5;
            end
            rx_bits = dec;
        else
            rx_bits = rx_bits(1:nb);
        end

        bit_errors = bit_errors + sum(rx_bits ~= binary_data(bit_idx-nb:bit_idx-1));
    end

    BER(snr_i) = bit_errors/no_bits;
end
end
