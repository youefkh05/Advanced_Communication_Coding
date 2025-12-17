%% Q6
% ===============================================================
% ====================== QUESTION 6 =============================
% ===============================================================
function Q6
% BPSK with (7,4) Hamming Code over AWGN (Hard Decision)

fprintf('Q6: BPSK with (7,4) Hamming Coding\n');
fprintf('---------------------------------\n');
pause(3);

%% ================= PARAMETERS ===============================
EbN0_dB = -3:1:10;
EbN0_linear = 10.^(EbN0_dB/10);
Eb = 1;

N_bits = 2e5;

% Hamming (7,4)
n = 7;
k = 4;
CodeRate = k/n;

% Ensure multiple of k
N_bits = ceil(N_bits/k)*k;

%% ================= UNCODED BPSK (THEORY) ====================
BER_uncoded_theory = 0.5 * erfc(sqrt(EbN0_linear));

%% ================= HAMMING (7,4) SIMULATION =================
BER_hamming_sim = zeros(size(EbN0_dB));

% Generate information bits
info_bits = randi([0 1], N_bits, 1);

% Encode
msg_words = reshape(info_bits, k, []).';
code_words = encode(msg_words, n, k, 'hamming/binary');

coded_bits = code_words.';
coded_bits = coded_bits(:);

% Energy scaling (same Eb per information bit)
Ec = Eb * CodeRate;
tx_amp = sqrt(Ec);

% BPSK modulation
tx_symbols = tx_amp * (2*coded_bits - 1);

for i = 1:length(EbN0_dB)

    % Noise variance based on Eb
    N0 = Eb / EbN0_linear(i);
    sigma = sqrt(N0/2);

    % AWGN
    noise = sigma * randn(size(tx_symbols));
    rx_symbols = tx_symbols + noise;

    % Hard decision
    rx_coded_bits = rx_symbols > 0;

    % Decode
    rx_code_words = reshape(rx_coded_bits, n, []).';
    rx_decoded_words = decode(rx_code_words, n, k, 'hamming/binary');

    rx_info_bits = rx_decoded_words.';
    rx_info_bits = rx_info_bits(:);

    % BER
    BER_hamming_sim(i) = mean(rx_info_bits ~= info_bits);
end

%% ===================== PLOTTING ==============================
fig = figure; hold on;

semilogy(EbN0_dB, BER_uncoded_theory, ...
    'k-', 'LineWidth', 2, ...
    'DisplayName','Uncoded BPSK (Theory)');

semilogy(EbN0_dB, BER_hamming_sim, ...
    'go--', ...
     'MarkerSize', 7, ...
     'LineWidth', 1.5,...
    'DisplayName','Hamming (7,4) Hard Decision');

grid on; grid minor;
xlabel('E_b/N_0 (dB)','FontWeight','bold');
ylabel('BER','FontWeight','bold');
title('BER Performance: Uncoded vs Hamming (7,4) Coded BPSK');
legend('Location','southwest');

set(gca,'YScale','log');
ylim([1e-5 1]);
xlim([-3 10]);

%% ================= TEXT BOX =================
annotation(fig,'textbox', ...
    [0.15 0.18 0.35 0.18], ...
    'String',{ ...
        'Code: Hamming (7,4)', ...
        'Minimum Distance d_{min} = 3', ...
        'Hard-decision decoding', ...
        'Same E_b per information bit'}, ...
    'FitBoxToText','on', ...
    'BackgroundColor','white');
 %% ===================== SAVE FIGURE ===========================
 save_figure_png(fig, ...
        'Q6_(7,4)_Hamming_code', ...
        'figures');

%% ================= REPORT ANSWERS =================
fprintf('(c) Minimum distance of (7,4) Hamming code: d_min = 3\n');
fprintf('(d) Recommendation:\n');
fprintf('     Yes for BER reduction at low SNR.\n');
fprintf('     No if bandwidth or transmission time is critical.\n');
end
