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
Nbits = 110000;                 % Number of bits

%%  BPSK Simulation
[BER_theory, BER_sim] = bpsk_uncoded_ber(EbN0_dB, Nbits, Eb)

%% ===================== PLOTTING ==============================
fig = plot_bpsk_ber(EbN0_dB, BER_theory, BER_sim);

% --- Save figure ---
save_figure_png(fig, ...
    'Q3_Uncoded_BPSK_AWGN', ...
    'results/Q3');





%%          Functions
%%  BPSK
function [BER_theory, BER_sim] = bpsk_uncoded_ber(EbN0_dB, Nbits, Eb)
% BPSK_UNCODED_BER
% Computes theoretical and simulated BER for uncoded BPSK over AWGN
%
% Inputs:
%   EbN0_dB : Eb/N0 values in dB (vector)
%   Nbits   : Number of bits
%   Eb      : Energy per bit
%
% Outputs:
%   BER_theory : Theoretical BER
%   BER_sim    : Simulated BER

    % === Convert Eb/N0 to linear scale ===
    EbN0_lin = 10.^(EbN0_dB/10);

    % === BPSK amplitude ===
    A = sqrt(Eb);

    % === THEORETICAL BER ===
    BER_theory = 0.5 * erfc(sqrt(EbN0_lin));

    % === MONTE-CARLO SIMULATION ===
    BER_sim = zeros(size(EbN0_dB));

    % Generate random bits
    tx_bits = randi([0 1], Nbits, 1);

    % BPSK modulation: 0 → -A, 1 → +A
    tx_symbols = A * (2*tx_bits - 1);

    % Loop over Eb/N0 values
    for k = 1:length(EbN0_dB)

        % Noise standard deviation
        sigma = sqrt((Eb/2) / EbN0_lin(k));

        % AWGN noise
        noise = sigma * randn(Nbits,1);

        % Received signal
        rx_symbols = tx_symbols + noise;

        % Hard decision detection
        rx_bits = rx_symbols > 0;

        % BER computation
        BER_sim(k) = mean(rx_bits ~= tx_bits);
    end
end
%%  Save Figure
function save_figure_png(figHandle, figName, savePath)
% SAVE_FIGURE_PNG
% Saves a MATLAB figure as PNG with proper formatting
%
% Inputs:
%   figHandle : handle to figure
%   figName   : string (figure title & filename)
%   savePath  : string (directory path)

    % --- Input checks ---
    if ~isvalid(figHandle)
        error('Invalid figure handle.');
    end

    if ~isfolder(savePath)
        mkdir(savePath);
    end

    % --- Set figure properties ---
    figHandle.Name = figName;
    figHandle.NumberTitle = 'off';

    % --- Build full file path ---
    fileName = fullfile(savePath, [figName '.png']);

    % --- Save figure ---
    exportgraphics(figHandle, fileName, 'Resolution', 300);

    fprintf('Figure saved successfully:\n%s\n', fileName);
end

%% Plot BPSK
function fig = plot_bpsk_ber(EbN0_dB, BER_theory, BER_sim)
% PLOT_BPSK_BER
% Plots theoretical and simulated BER for uncoded BPSK over AWGN
%
% Inputs:
%   EbN0_dB     : Eb/N0 values in dB
%   BER_theory : theoretical BER
%   BER_sim    : simulated BER
%
% Output:
%   fig        : figure handle

    fig = figure;

    % --- Theoretical BER (solid blue line) ---
    semilogy(EbN0_dB, BER_theory, ...
        'b-', 'LineWidth', 2);
    hold on;

    % --- Simulated BER (red circles with connecting dashed line) ---
    semilogy(EbN0_dB, BER_sim, ...
        'ro--', ...
        'MarkerSize', 7, ...
        'LineWidth', 1.5);

    grid on;
    grid minor;

    xlabel('E_b/N_0 (dB)', ...
        'FontSize', 12, ...
        'FontWeight','bold');

    ylabel('Bit Error Rate (BER)', ...
        'FontSize', 12, ...
        'FontWeight','bold');

    title('Figure 1: Uncoded BPSK over AWGN', ...
        'FontSize', 14, ...
        'FontWeight','bold');

    legend('Theoretical BER', 'Simulated BER', ...
        'Location','southwest');

    ylim([1e-5 1]);
    set(gca, 'FontSize', 11);
end
