%% Q7
% ===============================================================
% ====================== QUESTION 7 =============================
% ===============================================================
function Q7
% BPSK and QPSK with Hamming (15,11) Coding over AWGN

fprintf('Q7: Hamming (15,11) with BPSK and QPSK\n');
fprintf('-------------------------------------\n');
pause(3);

%% ================= PARAMETERS ===============================
EbN0_dB = -3:1:10;
EbN0_linear = 10.^(EbN0_dB/10);
Eb = 1;

N_bits_target = 6.6e5;

% Hamming (15,11)
n = 15;
k = 11;
R_code = k/n;

% Modulation parameters
k_mod_BPSK = 1;
k_mod_QPSK = 2;

% Ensure valid length
N_bits = ceil(N_bits_target/k)*k;
info_bits = randi([0 1], N_bits, 1);

%% ================= ENCODING ================================
msg_words = reshape(info_bits, k, []).';
code_words = encode(msg_words, n, k, 'hamming/binary');
coded_bits = code_words.';
coded_bits = coded_bits(:);
N_coded_bits = length(coded_bits);

%% ================= UNCODED BPSK (THEORY) ===================
BER_uncoded_theory = 0.5 * erfc(sqrt(EbN0_linear));

%% ================= CODED BPSK ==============================
BER_hamming_bpsk = zeros(size(EbN0_dB));

Ec_bpsk = Eb * R_code;
A_bpsk = sqrt(Ec_bpsk);
tx_symbols_bpsk = A_bpsk * (2*coded_bits - 1);

for i = 1:length(EbN0_dB)

    N0 = Eb / EbN0_linear(i);
    sigma = sqrt(N0/2);

    noise = sigma * randn(size(tx_symbols_bpsk));
    rx = tx_symbols_bpsk + noise;

    rx_bits = rx > 0;
    rx_code_words = reshape(rx_bits, n, []).';
    rx_decoded = decode(rx_code_words, n, k, 'hamming/binary');

    rx_info_bits = rx_decoded.';
    rx_info_bits = rx_info_bits(:);

    BER_hamming_bpsk(i) = mean(rx_info_bits ~= info_bits);
end

%% ================= CODED QPSK ==============================
BER_hamming_qpsk = zeros(size(EbN0_dB));

% Padding for QPSK
N_coded_bits_qpsk = ceil(N_coded_bits/2)*2;
coded_bits_qpsk = [coded_bits; zeros(N_coded_bits_qpsk - N_coded_bits,1)];
N_symbols_qpsk = N_coded_bits_qpsk/2;

% Energy per QPSK symbol
Es_qpsk = Eb * (k_mod_QPSK * R_code);
A_qpsk = sqrt(Es_qpsk)/sqrt(2);

bit_pairs = reshape(coded_bits_qpsk, 2, []).';
I = 2*bit_pairs(:,1) - 1;
Q = 2*bit_pairs(:,2) - 1;
tx_symbols_qpsk = A_qpsk * (I + 1i*Q);

for i = 1:length(EbN0_dB)

    N0 = Eb / EbN0_linear(i);
    sigma = sqrt(N0/2);

    noise = sigma * (randn(size(tx_symbols_qpsk)) + 1i*randn(size(tx_symbols_qpsk)));
    rx = tx_symbols_qpsk + noise;

    rx_I = real(rx) > 0;
    rx_Q = imag(rx) > 0;

    rx_bits = [rx_I rx_Q].';
    rx_bits = rx_bits(:);
    rx_bits = rx_bits(1:N_coded_bits);

    rx_code_words = reshape(rx_bits, n, []).';
    rx_decoded = decode(rx_code_words, n, k, 'hamming/binary');

    rx_info_bits = rx_decoded.';
    rx_info_bits = rx_info_bits(:);

    BER_hamming_qpsk(i) = mean(rx_info_bits ~= info_bits);
end

%% ===================== PLOTTING ==============================
fig = figure; hold on;

semilogy(EbN0_dB, BER_uncoded_theory, 'k-', 'LineWidth', 2, ...
    'DisplayName','Uncoded BPSK (Theory)');

semilogy(EbN0_dB, BER_hamming_bpsk, 'go--', 'LineWidth', 1.5, ...
    'MarkerSize', 7, 'DisplayName','Hamming (15,11) BPSK');

semilogy(EbN0_dB, BER_hamming_qpsk, 'bo--', 'LineWidth', 1.5, ...
    'MarkerSize', 7, 'DisplayName','Hamming (15,11) QPSK');

grid on; grid minor;
xlabel('E_b/N_0 (dB)','FontWeight','bold');
ylabel('BER','FontWeight','bold');
title('BER Performance: Hamming (15,11) Coded BPSK vs QPSK');
legend('Location','southwest');
set(gca,'YScale','log');
ylim([1e-5 1]);
xlim([-3 10]);

%% ================= TEXT BOX =================
annotation(fig,'textbox', ...
    [0.15 0.18 0.45 0.18], ...
    'String',{ ...
        'Hamming (15,11)', ...
        'Hard-decision decoding', ...
        'QPSK achieves same BER as BPSK', ...
        'QPSK transmits 2 bits/symbol → higher data rate'}, ...
    'FitBoxToText','on', ...
    'BackgroundColor','white');
 %% ===================== SAVE FIGURE ===========================
 save_figure_png(fig, ...
        'Q7_QPSK_BPSK_(15,11)_Hamming code', ...
        'figures');

%% ================= REPORT ANSWER (7.1.e) ====================
fprintf('(e) Recommendation:\n');
fprintf(' QPSK is preferred.\n');
fprintf(' Same BER as BPSK with double transmission rate.\n');
fprintf(' More spectrally efficient with no BER penalty.\n');
end
