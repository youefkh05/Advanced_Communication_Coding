clear;
clc;
close all;

%% ================= PARAMETERS =================
Nbits = 1e5;
Eb = 1;
R  = 5;                       % Repetition factor

EbN0_dB  = -3:1:10;
EbN0_lin = 10.^(EbN0_dB/10);

% BER storage
BER_BPSK_unc  = zeros(size(EbN0_dB));
BER_BPSK_rep  = zeros(size(EbN0_dB));
BER_QPSK_unc  = zeros(size(EbN0_dB));
BER_QPSK_rep  = zeros(size(EbN0_dB));
BER_16QAM_unc = zeros(size(EbN0_dB));
BER_16QAM_rep = zeros(size(EbN0_dB));

%% ================= MAIN Eb/N0 LOOP =================
for idx = 1:length(EbN0_dB)

    N0 = Eb / EbN0_lin(idx);

    %% ==================================================
    %% ======================= BPSK =====================
    %% ==================================================
    bk = randi([0 1], Nbits, 1);

    % ----- Uncoded BPSK -----
    xk = sqrt(Eb) * (2*bk - 1);

    h = (randn(Nbits,1) + 1j*randn(Nbits,1)) / sqrt(2);
    n = sqrt(N0/2) * (randn(Nbits,1) + 1j*randn(Nbits,1));

    y = h .* xk + n;
    y_eq = y ./ h;
    bk_hat = real(y_eq) > 0;

    BER_BPSK_unc(idx) = mean(bk_hat ~= bk);

    % ----- Repetition BPSK -----
    bk_rep = repelem(bk, R);
    xk_rep = sqrt(Eb) * (2*bk_rep - 1);

    h = (randn(length(xk_rep),1) + 1j*randn(length(xk_rep),1)) / sqrt(2);
    n = sqrt(N0/2) * (randn(length(xk_rep),1) + 1j*randn(length(xk_rep),1));

    y = h .* xk_rep + n;
    y_eq = y ./ h;

    bk_hat_rep = real(y_eq) > 0;
    bk_hat_mat = reshape(bk_hat_rep, R, []);
    bk_hat = sum(bk_hat_mat,1) >= ceil(R/2);

    BER_BPSK_rep(idx) = mean(bk_hat.' ~= bk);

    %% ==================================================
    %% ======================= QPSK =====================
    %% ==================================================
    bits = bk(1:2*floor(length(bk)/2));
    b = reshape(bits,2,[]).';

    xk = sqrt(Eb/2) * ((2*b(:,1)-1) + 1j*(2*b(:,2)-1));

    h = (randn(length(xk),1) + 1j*randn(length(xk),1)) / sqrt(2);
    n = sqrt(N0/2) * (randn(length(xk),1) + 1j*randn(length(xk),1));

    y = h .* xk + n;
    y_eq = y ./ h;

    b1 = real(y_eq) > 0;
    b2 = imag(y_eq) > 0;

    bk_hat = reshape([b1 b2].',[],1);
    BER_QPSK_unc(idx) = mean(bk_hat ~= bits);

    % ----- Repetition QPSK -----
    bits_rep = repelem(bits, R);
    b = reshape(bits_rep,2,[]).';

    xk = sqrt(Eb/2) * ((2*b(:,1)-1) + 1j*(2*b(:,2)-1));

    h = (randn(length(xk),1) + 1j*randn(length(xk),1)) / sqrt(2);
    n = sqrt(N0/2) * (randn(length(xk),1) + 1j*randn(length(xk),1));

    y = h .* xk + n;
    y_eq = y ./ h;

    b1 = real(y_eq) > 0;
    b2 = imag(y_eq) > 0;

    bk_hat_rep = reshape([b1 b2].',[],1);
    bk_hat_mat = reshape(bk_hat_rep, R, []);
    bk_hat = sum(bk_hat_mat,1) >= ceil(R/2);

    BER_QPSK_rep(idx) = mean(bk_hat.' ~= bits);

    %% ==================================================
    %% ===================== 16-QAM =====================
    %% ==================================================
    bits = bk(1:4*floor(length(bk)/4));
    b = reshape(bits,4,[]).';

    I = (2*b(:,1)-1) .* (2 - (2*b(:,3)));
    Q = (2*b(:,2)-1) .* (2 - (2*b(:,4)));
    xk = (I + 1j*Q) / sqrt(10);

    h = (randn(length(xk),1) + 1j*randn(length(xk),1)) / sqrt(2);
    n = sqrt(N0/2) * (randn(length(xk),1) + 1j*randn(length(xk),1));

    y = h .* xk + n;
    y_eq = y ./ h;

    I_hat = real(y_eq);
    Q_hat = imag(y_eq);

    b1 = I_hat > 0;
    b2 = Q_hat > 0;
    b3 = abs(I_hat) < 2;
    b4 = abs(Q_hat) < 2;

    bk_hat = reshape([b1 b2 b3 b4].',[],1);
    BER_16QAM_unc(idx) = mean(bk_hat ~= bits);

    % ----- Repetition 16-QAM -----
    bits_rep = repelem(bits, R);
    b = reshape(bits_rep,4,[]).';

    I = (2*b(:,1)-1) .* (2 - (2*b(:,3)));
    Q = (2*b(:,2)-1) .* (2 - (2*b(:,4)));
    xk = (I + 1j*Q) / sqrt(10);

    h = (randn(length(xk),1) + 1j*randn(length(xk),1)) / sqrt(2);
    n = sqrt(N0/2) * (randn(length(xk),1) + 1j*randn(length(xk),1));

    y = h .* xk + n;
    y_eq = y ./ h;

    I_hat = real(y_eq);
    Q_hat = imag(y_eq);

    b1 = I_hat > 0;
    b2 = Q_hat > 0;
    b3 = abs(I_hat) < 2;
    b4 = abs(Q_hat) < 2;

    bk_hat_rep = reshape([b1 b2 b3 b4].',[],1);
    bk_hat_mat = reshape(bk_hat_rep, R, []);
    bk_hat = sum(bk_hat_mat,1) >= ceil(R/2);

    BER_16QAM_rep(idx) = mean(bk_hat.' ~= bits);
end

%% ===================== FIGURES =====================

%% ===================== FIGURES =====================

% ---- BPSK ----
fig = plot_rayleigh(EbN0_dB, BER_BPSK_unc, BER_BPSK_rep, Nbits, 'BPSK');
save_figure_png(fig, ...
    'Q2_BPSK_Rayleigh_Uncoded_vs_Repetition', ...
    'figures');

% ---- QPSK ----
fig = plot_rayleigh(EbN0_dB, BER_QPSK_unc, BER_QPSK_rep, Nbits, 'QPSK');
save_figure_png(fig, ...
    'Q2_QPSK_Rayleigh_Uncoded_vs_Repetition', ...
    'figures');

% ---- 16-QAM ----
fig = plot_rayleigh(EbN0_dB, BER_16QAM_unc, BER_16QAM_rep, Nbits, '16-QAM');
save_figure_png(fig, ...
    'Q2_16QAM_Rayleigh_Uncoded_vs_Repetition', ...
    'figures');

% ---- Combined Uncoded Comparison ----
fig = figure;

semilogy(EbN0_dB, BER_BPSK_unc, 'r-o', 'LineWidth', 1.8); hold on;
semilogy(EbN0_dB, BER_QPSK_unc, 'b-s', 'LineWidth', 1.8);
semilogy(EbN0_dB, BER_16QAM_unc,'k-^', 'LineWidth', 1.8);

grid on; grid minor;

legend('BPSK','QPSK','16-QAM','Location','southwest');
xlabel('E_b / N_0 (dB)','FontWeight','bold');
ylabel('BER','FontWeight','bold');
title('Uncoded Modulation Comparison over Rayleigh Flat Fading Channel', ...
      'FontWeight','bold');

set(gca,'YScale','log','FontSize',11);

save_figure_png(fig, ...
    'Q2_Uncoded_BPSK_QPSK_16QAM_Comparison', ...
    'figures');


%% ========================= Functions ========================
%% Plot BER
function fig = plot_rayleigh(EbN0_dB, BER_uncoded, BER_rep, Nbits, modType)
% PLOT_Q2_BPSK_RAYLEIGH
% Plots uncoded and repetition-coded transmission over Rayleigh flat fading

    fig = figure;

    semilogy(EbN0_dB, BER_uncoded, ...
        'ro--', 'LineWidth', 1.8, 'MarkerSize', 7);
    hold on;

    semilogy(EbN0_dB, BER_rep, ...
        'bs--', 'LineWidth', 1.8, 'MarkerSize', 7);

    grid on;
    grid minor;

    xlabel('E_b / N_0 (dB)', ...
        'FontSize', 12, 'FontWeight', 'bold');

    ylabel('Bit Error Rate (BER)', ...
        'FontSize', 12, 'FontWeight', 'bold');

    title([modType ' over Rayleigh Flat Fading Channel'], ...
        'FontSize', 14, 'FontWeight', 'bold');

    legend( ...
        ['Uncoded ' modType], ...
        [modType ' with Repetition (Rate 1/5)'], ...
        'Location','southwest');

    set(gca, 'FontSize', 11);
    set(gca, 'YScale', 'log');

    % Info box
    annotation(fig,'textbox', ...
        [0.15 0.18 0.38 0.18], ...
        'String',{ ...
            ['Modulation: ' modType], ...
            'Channel: Rayleigh Flat Fading', ...
            'Coding: Repetition (Rate 1/5)', ...
            sprintf('# bits: %d', Nbits)}, ...
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
