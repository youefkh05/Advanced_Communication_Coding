%% ============================================================
%  Problem 5
% ============================================================
clc;
clear;
close all;
% BPSK with Repetition-3 Coding (Soft Decision)

    %% ================= PARAMETERS ===============================
    EbN0_dB = -3:1:10;
    EbN0_lin = 10.^(EbN0_dB/10);
    Eb = 1;
    A = sqrt(Eb);
    Nbits = 110000;
    R = 3; % repetition factor

    %% ================= UNCODED (THEORETICAL) ====================
    BER_uncoded = 0.5 * erfc(sqrt(EbN0_lin));

    %% ================= SIMULATION ===============================
    BER_same_Etx = zeros(size(EbN0_dB));
    BER_same_Einfo = zeros(size(EbN0_dB));

    % Generate information bits
    bits = randi([0 1], Nbits, 1);

    % Repetition coding
    coded_bits = repelem(bits, R);

    %% ===== Case 1: Same Energy per Transmitted Bit (Hard decision) =====
    tx_symbols = A * (2*coded_bits - 1);

    for k = 1:length(EbN0_dB)
        sigma = sqrt((Eb/2) / EbN0_lin(k));
        noise = sigma * randn(length(tx_symbols),1);
        rx = tx_symbols + noise;

        % Hard decision
        rx_bits = rx > 0;

        % Majority voting
        rx_matrix = reshape(rx_bits, R, []);
        decoded_bits = sum(rx_matrix,1) >= 2;

        BER_same_Etx(k) = mean(decoded_bits.' ~= bits);
    end

    %% ===== Case 2: Same Energy per Information Bit (Soft decision) =====
    A_info = A / sqrt(R);
    tx_symbols = A_info * (2*coded_bits - 1);

    for k = 1:length(EbN0_dB)
        sigma = sqrt((Eb/2) / EbN0_lin(k));
        noise = sigma * randn(length(tx_symbols),1);
        rx = tx_symbols + noise;

        % SOFT decision demapper
        rx_matrix = reshape(rx, R, []);

        % Averaging decoder
        decoded_bits = mean(rx_matrix,1) > 0;

        BER_same_Einfo(k) = mean(decoded_bits.' ~= bits);
    end

    %% ===================== PLOTTING ==============================
    fig = plot_q5_ber(EbN0_dB, ...
        BER_uncoded, ...
        BER_same_Etx, ...
        BER_same_Einfo, ...
        Nbits);

    %% ===================== SAVE FIGURE ===========================
    save_figure_png(fig, ...
        'Q5_BPSK_Repetition3_SoftDecision', ...
        'results/Q5');


%% Plot Q5
function fig = plot_q5_ber(EbN0_dB, BER_uncoded, BER_Etx, BER_Einfo, Nbits)

    fig = figure;

    semilogy(EbN0_dB, BER_uncoded, ...
        'k-', 'LineWidth', 2.5); 
    hold on;

    semilogy(EbN0_dB, BER_Etx, ...
        'ro--', 'LineWidth', 1.5, 'MarkerSize',6);

    semilogy(EbN0_dB, BER_Einfo, ...
        'bs--', 'LineWidth', 1.5, 'MarkerSize',6);

    grid on; grid minor;

    xlabel('E_b/N_0 (dB)', 'FontSize',12,'FontWeight','bold');
    ylabel('Bit Error Rate (BER)', 'FontSize',12,'FontWeight','bold');

    title('Figure 3: BPSK with Repetition-3 (Soft Decision)', ...
        'FontSize',14,'FontWeight','bold');

    legend( ...
        'Uncoded BPSK (Theoretical)', ...
        'Repetition-3 (Same E per Tx Bit, Hard)', ...
        'Repetition-3 (Same E per Info Bit, Soft)', ...
        'Location','southwest');

    ylim([1e-6 1]);
    set(gca,'FontSize',11);

    % Info box
    annotation(fig,'textbox', ...
        [0.15 0.18 0.38 0.18], ...
        'String',{ ...
            'Modulation: BPSK', ...
            'Coding: Repetition-3', ...
            'Decoder: Soft Decision (Averaging)', ...
            sprintf('Monte-Carlo bits: %d', Nbits)}, ...
        'FitBoxToText','on', ...
        'BackgroundColor','white', ...
        'EdgeColor','black', ...
        'FontSize',10);
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
