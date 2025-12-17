%% ============================================================
%  Problem 3: Uncoded BPSK over AWGN Channel
% ============================================================
clc;
clear;
close all;

%% ================= PARAMETERS ===============================
EbN0_dB = -3:1:10;              % Eb/No range in dB
EbN0_lin = 10.^(EbN0_dB/10);    % Linear scale
Eb = 1;                         % Energy per bit
A = sqrt(Eb);                   % BPSK amplitude
Nbits = 110000;                    % Number of Monte-Carlo bits

%% ================= THEORETICAL BER ==========================
BER_theory = 0.5 * erfc(sqrt(EbN0_lin));

%% ================= MONTE-CARLO SIMULATION ===================
BER_sim = zeros(size(EbN0_dB));

% Generate random information bits
tx_bits = randi([0 1], Nbits, 1);

% BPSK mapping: 0 → -A , 1 → +A
tx_symbols = A * (2*tx_bits - 1);

for k = 1:length(EbN0_dB)
    
    % Noise variance
    sigma = sqrt((Eb/2) / EbN0_lin(k));
    
    % AWGN
    noise = sigma * randn(Nbits,1);
    rx_symbols = tx_symbols + noise;
    
    % Hard decision detector
    rx_bits = rx_symbols > 0;
    
    % BER computation
    BER_sim(k) = mean(rx_bits ~= tx_bits);
end

%% ===================== PLOTTING ==============================
figure;
plot(EbN0_dB, BER_theory, 'k-', 'LineWidth', 2); hold on;
plot(EbN0_dB, BER_sim, 'ko', 'MarkerSize', 7, 'LineWidth', 1.5);

grid on; grid minor;
xlabel('E_b/N_0 (dB)', 'FontSize', 12, 'FontWeight','bold');
ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'FontWeight','bold');
title('Figure 1: Uncoded BPSK over AWGN', 'FontSize', 14, 'FontWeight','bold');
legend('Theoretical BER', 'Simulated BER', 'Location','southwest');

ylim([1e-5 1]);
set(gca,'YScale','log','FontSize',11);
