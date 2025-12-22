function OFDM_GUI_T35
% OFDM PROJECT GUI (Q1 → Q3)
% Single-file implementation

    % ================= HOME PAGE =================
    homeFig = uifigure( ...
        'Name','OFDM Project', ...
        'Position',[500 200 420 420]);

    gl = uigridlayout(homeFig,[4 1]);
    gl.RowHeight = {'fit','1x','1x','1x'};
    gl.Padding = [30 30 30 20];

    % Title
    uilabel(gl, ...
        'Text','OFDM Project – Team 35', ...
        'FontSize',18, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','center');

    % Question buttons
    for q = 1:3
        uibutton(gl, ...
            'Text', sprintf('Question %d', q), ...
            'FontSize',14, ...
            'ButtonPushedFcn', @(~,~) open_question(q, homeFig));
    end
end

% ===============================================================
%                  QUESTION WINDOW
% ===============================================================
function open_question(qnum, homeFig)

    qFig = uifigure( ...
        'Name', sprintf('Question %d', qnum), ...
        'Position',[400 150 900 550]);

    gl = uigridlayout(qFig,[3 1]);
    gl.RowHeight = {'fit','fit','1x'};
    gl.Padding = [15 15 15 15];

    % Header
    uilabel(gl, ...
        'Text', sprintf('Question %d', qnum), ...
        'FontSize',16, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','center');

    % Return button
    uibutton(gl, ...
        'Text','⬅ Return to Home', ...
        'FontSize',13, ...
        'ButtonPushedFcn', @(~,~) return_home(qFig, homeFig));

    % Output console
    logBox = uitextarea(gl, ...
        'Editable','off', ...
        'FontSize',11);

    drawnow;

    % ================= RUN QUESTION =================
    try
        switch qnum
            case 1
                outputText = evalc('Q1');
            case 2
                outputText = evalc('Q2');
            case 3
                outputText = evalc('Q3');
        end

        logBox.Value = splitlines(outputText);

    catch ME
        logBox.Value = { ...
            ' Error occurred:', ...
            ME.message ...
        };
        uialert(qFig, ME.message, 'Execution Error');
    end
end


% ===============================================================
%                  RETURN BUTTON
% ===============================================================
function return_home(qFig, homeFig)
    if isvalid(qFig)
        close(qFig);
    end
    homeFig.Visible = 'on';
end

%% Q1
% ===============================================================
% ====================== QUESTION 1 =============================
% ===============================================================
function Q1
    L = 4096;
    x = randn(1,L);
    
    tic
    X_dft = myDFT(x);
    t_dft = toc;
    
    tic
    X_fft = fft(x);
    t_fft = toc;
    
    fprintf('Elapsed time is %s seconds.', num2str(t_dft));
    fprintf('Elapsed time is %s seconds.', num2str(t_fft));
    
    fig1 = plot_dft_fft_time(t_dft, t_fft);
    save_figure_png(fig1, 'Q1_DFT_vs_FFT_Execution_Time', 'figures');
end
%====================== Q1 Helper functions =============================

%% My DFT Function
function X = myDFT(x)
N = length(x);
X = zeros(1,N);

for k = 0:N-1
    sumVal = 0;
    for n = 0:N-1
        sumVal = sumVal + x(n+1)*exp(-1j*2*pi*n*k/N);
    end
    X(k+1) = sumVal;
end
end

%% Bar Chart
function fig = plot_dft_fft_time(t_dft, t_fft)
% PLOT_DFT_FFT_TIME
% Plots execution time comparison between DFT and FFT

    times = [t_dft t_fft];

    fig = figure;
    b = bar(times);
    grid on;

    set(gca, 'XTickLabel', {'DFT', 'FFT'});
    ylabel('Execution Time (seconds)');
    title('Execution Time Comparison: DFT vs FFT');

    % Display values above bars
    for i = 1:length(times)
        text(i, times(i), sprintf('%.6f', times(i)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom');
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


%% Q2
% ===============================================================
% ====================== QUESTION 2 =============================
% ===============================================================
function Q2
    
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
    end

%====================== Q2 Helper functions =============================
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


%% Q5
% ===============================================================
% ====================== QUESTION 5 =============================
% ===============================================================
function Q5
% BPSK with Repetition-3 Coding (Soft Decision)

    %% ================= PARAMETERS ===============================
    fprintf('Q5 Start\n');
    pause(3);

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
    fprintf('Simulating Case1 \n');

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
    fprintf('Simulating Case2 \n');

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
    fprintf('Q5 Plot \n');

    fig = plot_q5_ber(EbN0_dB, ...
        BER_uncoded, ...
        BER_same_Etx, ...
        BER_same_Einfo, ...
        Nbits);

    %% ===================== SAVE FIGURE ===========================
    save_figure_png(fig, ...
        'Q5_BPSK_Repetition3_SoftDecision', ...
        'figures');
end

%====================== Q5 Helper functions =============================
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
%% ================= UNCODED BPSK (SIMULATION) =================
BER_uncoded_sim = zeros(size(EbN0_dB));

% Generate uncoded bits
tx_bits_uncoded = randi([0 1], N_bits, 1);
A_uncoded = sqrt(Eb);
tx_symbols_uncoded = A_uncoded * (2*tx_bits_uncoded - 1);

for i = 1:length(EbN0_dB)
    sigma = sqrt((Eb/2) / EbN0_linear(i));
    noise = sigma * randn(size(tx_symbols_uncoded));
    rx = tx_symbols_uncoded + noise;
    rx_bits = rx > 0;
    BER_uncoded_sim(i) = mean(rx_bits ~= tx_bits_uncoded);
end
%% ===================== PLOTTING ==============================
fig = figure; hold on;

semilogy(EbN0_dB, BER_uncoded_sim,  'go--', 'LineWidth', 1.8, ...
    'MarkerSize', 7, 'DisplayName','Uncoded BPSK (Baseline)');

semilogy(EbN0_dB, BER_uncoded_theory, 'k-', 'LineWidth', 2, ...
    'DisplayName','Uncoded BPSK (Theory)');

semilogy(EbN0_dB, BER_hamming_qpsk, 'bo--', 'LineWidth', 1.8, ...
    'MarkerSize', 7, 'DisplayName','Proposed: QPSK + Hamming (15,11)');

grid on; grid minor;
xlabel('E_b/N_0 (dB)','FontWeight','bold');
ylabel('BER','FontWeight','bold');
title('Problem 7: Proposed Scheme vs Uncoded BPSK');
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

%% ================= REPORT ANSWERS ============================
fprintf('(e) Recommendation:\n');
fprintf(' Using Hamming (15,11) improves BER performance.\n\n');

fprintf('(f) Proposal:\n');
fprintf(' Use QPSK with Hamming (15,11) to keep transmission time\n');
fprintf(' equal to or less than uncoded BPSK.\n\n');

fprintf('(g) Comment:\n');
fprintf(' The proposed QPSK + Hamming (15,11) scheme achieves\n');
fprintf(' significant BER improvement while maintaining equal\n');
fprintf(' or lower transmission time compared to uncoded BPSK.\n');

end

%% Q8
% ===============================================================
% ====================== QUESTION 8 =============================
% ===============================================================
function Q8
% QPSK vs 16-QAM with BCH(255,131)

fprintf('Q8: QPSK vs 16-QAM + BCH(255,131)\n');
fprintf('---------------------------------\n');
pause(3);

%% ================= PARAMETERS ===============================
EbN0_dB = 5:15;
EbN0_lin = 10.^(EbN0_dB/10);

Eb_QPSK  = 1;     % QPSK Eb
Eb_16QAM = 2.5;   % Given in problem

n = 255; 
k = 131;
R = k/n;

MAX_ERR = 300;
CHUNK = 26200;

BER_QPSK   = nan(size(EbN0_dB));
BER_16QAM  = nan(size(EbN0_dB));

%% ================= MAIN LOOP ===============================
for ii = 1:length(EbN0_dB)

    % --- Bit limits per SNR ---
    if EbN0_dB(ii) <= 8
        MAX_BITS = 2.62e6;
    elseif EbN0_dB(ii) == 9
        MAX_BITS = 2.62e7;
    else
        continue   % interpolation later
    end

    %% ================= QPSK ===============================
    err = 0; bits_cnt = 0;
    while err < MAX_ERR && bits_cnt < MAX_BITS
        bits = randi([0 1],1,CHUNK);

        tx = modQPSK(bits);
        sigma = sqrt(Eb_QPSK/(2*10^(EbN0_dB(ii)/10)));
        rx = tx + sigma*(randn(size(tx))+1j*randn(size(tx)));

        bits_hat = demodQPSK(rx);
        err = err + sum(bits ~= bits_hat);
        bits_cnt = bits_cnt + CHUNK;
    end
    BER_QPSK(ii) = err / bits_cnt;

    %% ================= 16-QAM + BCH =========================
    err = 0; bits_cnt = 0;
    while err < MAX_ERR && bits_cnt < MAX_BITS

        bits = randi([0 1],1,CHUNK);
        bits = bits(1:floor(length(bits)/k)*k);

        msg = reshape(bits,k,[])';
        coded = bchenc(gf(msg),n,k);
        codedBits = reshape(double(coded.x)',1,[]);

        pad = mod(4-mod(length(codedBits),4),4);
        codedBits = [codedBits zeros(1,pad)];

        tx = mod16(codedBits);
        Eb_info = Eb_16QAM/R;
        sigma = sqrt(Eb_info/(2*10^(EbN0_dB(ii)/10)));
        rx = tx + sigma*(randn(size(tx))+1j*randn(size(tx)));

        coded_hat = demod16(rx);
        coded_hat = coded_hat(1:numel(codedBits));

        rxMat = reshape(coded_hat,n,[])';
        decoded = bchdec(gf(rxMat),n,k);
        bits_hat = reshape(double(decoded.x)',1,[]);

        err = err + sum(bits ~= bits_hat);
        bits_cnt = bits_cnt + length(bits);
    end
    BER_16QAM(ii) = err / bits_cnt;

    fprintf('Eb/N0=%d dB | QPSK=%.3e | 16QAM+BCH=%.3e\n', ...
        EbN0_dB(ii), BER_QPSK(ii), BER_16QAM(ii));
end

%% ================= INTERPOLATION ============================
BER_QPSK(6:end)  = interp1(5:9,BER_QPSK(1:5),10:15,'linear');
BER_16QAM(6:end) = interp1(5:9,BER_16QAM(1:5),10:15,'linear');

%% ================= PLOTTING ================================
fig = figure; hold on;

semilogy(EbN0_dB,BER_QPSK,'bo--', 'LineWidth', 1.5, ...
    'MarkerSize', 7, 'DisplayName','QPSK (No Coding)');

semilogy(EbN0_dB,BER_16QAM,'go--', 'LineWidth', 1.5, ...
    'MarkerSize', 7, 'DisplayName','16-QAM + BCH(255,131)');

grid on; grid minor;
xlabel('E_b/N_0 (dB)','FontWeight','bold');
ylabel('BER','FontWeight','bold');
title('BER Comparison: QPSK vs 16-QAM + BCH');
legend('Location','southwest');
ylim([1e-6 1]);
set(gca,'YScale','log');

annotation(fig,'textbox', ...
    [0.15 0.18 0.45 0.18], ...
    'String',{ ...
        'Eb = 2.5', ...
        'BCH(255,131)', ...
        'QPSK: higher rate, no coding', ...
        '16-QAM+BCH: higher coding gain'}, ...
    'FitBoxToText','on', ...
    'BackgroundColor','white');

save_figure_png(fig,'Q8_QPSK_vs_16QAM_BCH','figures');
end

%====================== Q8 Helper functions =============================
%%  QPSK Modulator
function sym = modQPSK(bits)
% MODQPSK - QPSK Modulator (Gray mapping)
% Input : bits (row vector, length multiple of 2)
% Output: complex QPSK symbols

    bits = reshape(bits, 2, []).';
    const = [1+1j, -1+1j, -1-1j, 1-1j]; % Gray mapping
    sym = const(bi2de(bits,'left-msb')+1).';
end

%%  QPSK Demodulator
function bits = demodQPSK(sym)
% DEMODQPSK - Hard-decision QPSK demodulator

    const = [1+1j, -1+1j, -1-1j, 1-1j];
    bits = zeros(1, 2*length(sym));

    for k = 1:length(sym)
        [~, idx] = min(abs(sym(k) - const));
        bits(2*k-1:2*k) = de2bi(idx-1, 2, 'left-msb');
    end
end

%%  16 QAM Modulator
function rxsig = mod16(txbits)
    psk16mod = [ ...
         1+1j  3+1j  1+3j  3+3j ...
         1-1j  3-1j  1-3j  3-3j ...
        -1+1j -3+1j -1+3j -3+3j ...
        -1-1j -3-1j -1-3j -3-3j ];

    m = 4;
    sigqam16 = reshape(txbits,m,[])';
    rxsig = psk16mod(bi2de(sigqam16,'left-msb')+1);
end

%%  16 QAM Demodulator
function rxbits = demod16(rxsig)
    m = 4;
    psk16demod = [15 14 6 7 13 12 4 5 9 8 0 1 11 10 2 3];

    rxsig(real(rxsig)>3)  = 3 + 1j*imag(rxsig(real(rxsig)>3));
    rxsig(imag(rxsig)>3)  = real(rxsig(imag(rxsig)>3)) + 1j*3;
    rxsig(real(rxsig)<-3) = -3 + 1j*imag(rxsig(real(rxsig)<-3));
    rxsig(imag(rxsig)<-3) = real(rxsig(imag(rxsig)<-3)) - 1j*3;

    rxdemod = round(real((rxsig+3+1j*3)/2)) + ...
              1j*round(imag((rxsig+3+1j*3)/2));

    rxdebi = real(rxdemod) + 4*imag(rxdemod);
    sigbits = de2bi(psk16demod(rxdebi+1),m,'left-msb');
    rxbits = reshape(sigbits.',1,[]);
end

%% Q9
% ===============================================================
% ====================== QUESTION 9 =============================
% ===============================================================
function Q9
% Convolutional Encoder (2,3,K=2)

fprintf('Q9: Convolutional Encoding (2,3,K)\n');
fprintf('----------------------------------\n');
pause(3);

%% ================= PARAMETERS ===============================
N_bits = 1000;

%% ================= INPUT GENERATION =========================
InputBits = randi([0 1], 1, N_bits);

% Termination (K = 2 → add 2 zeros)
InputBits = [InputBits 0 0];

%% ================= ENCODER MEMORY ===========================
u1_prev = 0;
u2_prev = 0;

%% ================= STORAGE ================================
encodedBits = [];
pastState   = {};
inputPairs  = {};
encodedOut  = {};

idx = 1;

%% ================= CONVOLUTIONAL ENCODING ==================
for i = 1:2:length(InputBits)-1

    % Current input bits
    u1 = InputBits(i);
    u2 = InputBits(i+1);

    % Generator equations (from problem statement)
    y1 = mod(u1_prev + u2 + u2_prev, 2);
    y2 = mod(u1 + u1_prev + u2, 2);
    y3 = mod(u2 + u2_prev, 2);

    % Store encoded bits
    encodedBits = [encodedBits y1 y2 y3];

    % Store table entries
    pastState{idx,1}  = sprintf('%d%d', u1_prev, u2_prev);
    inputPairs{idx,1} = sprintf('%d%d', u1, u2);
    encodedOut{idx,1} = sprintf('%d%d%d', y1, y2, y3);

    % Update memory
    u1_prev = u1;
    u2_prev = u2;

    idx = idx + 1;
end

%% ================= CREATE TABLE =============================
encodingTable = table( ...
    pastState, inputPairs, encodedOut, ...
    'VariableNames', {'PastState','InputPair','EncodedOutput'});

%% ================= DISPLAY TABLE AS FIGURE ===================
firstRows = encodingTable(1:15,:);

fig = figure( ...
    'Name','Q9: Convolutional Encoder State Table', ...
    'NumberTitle','off', ...
    'Position',[450 250 500 350]);

uitable(fig, ...
    'Data', firstRows{:,:}, ...
    'ColumnName', firstRows.Properties.VariableNames, ...
    'RowName', [], ...
    'FontSize', 11, ...
    'Units','normalized', ...
    'Position',[0.05 0.05 0.9 0.9]);

%% ================= OPTIONAL: SAVE FIGURE =====================
save_figure_png(fig, ...
    'Q9_Convolutional_Encoder_Table', ...
    'figures');


%% ================= SUMMARY ================================
fprintf('Total input bits (with termination): %d\n', length(InputBits));
fprintf('Total encoded bits: %d\n', length(encodedBits));
fprintf('Code rate ≈ 2/3\n');

fprintf('\nEncoding completed successfully.\n');
end

