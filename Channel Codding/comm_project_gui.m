function comm_project_gui
% DIGITAL COMMUNICATIONS PROJECT GUI (Q3 → Q9)
% Single-file implementation

    

    % ================= HOME PAGE =================
    homeFig = uifigure( ...
        'Name','Digital Communications Project', ...
        'Position',[500 200 420 500]);

    gl = uigridlayout(homeFig,[8 1]);
    gl.RowHeight = {'fit','1x','1x','1x','1x','1x','1x','1x'};
    gl.Padding = [20 20 20 20];

    % Title
    uilabel(gl, ...
        'Text','Digital Communications Project', ...
        'FontSize',18, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','center');

    % Question buttons
    for q = 3:9
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

    % Output console (replacement for Command Window)
    logBox = uitextarea(gl, ...
        'Editable','off', ...
        'FontSize',11);

    drawnow;

    % ================= RUN QUESTION WITH OUTPUT CAPTURE =================
    try
        switch qnum
            case 3
                outputText = evalc('Q3');
            case 4
                outputText = evalc('Q4');
            case 5
                outputText = evalc('Q5');
            case 6
                outputText = evalc('Q6');
            case 7
                outputText = evalc('Q7');
            case 8
                outputText = evalc('Q8');
            case 9
                outputText = evalc('Q9');
        end

        logBox.Value = splitlines(outputText);

    catch ME
        logBox.Value = { ...
            '❌ Error occurred:', ...
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

%% Q3
% ===============================================================
% ====================== QUESTION 3 =============================
% ===============================================================
function Q3
% Uncoded BPSK over AWGN


    %% ================= PARAMETERS ===============================
    EbN0_dB = -3:1:10;              % Eb/No range in dB
    EbN0_lin = 10.^(EbN0_dB/10);    % Linear scale
    Eb = 1;                         % Energy per bit
    A = sqrt(Eb);                   % BPSK amplitude
    Nbits = 110000;                 % Number of bits
    
    %%  BPSK Simulation
    [BER_theory, BER_sim] = bpsk_uncoded_ber(EbN0_dB, Nbits, Eb);
    
    %% ===================== PLOTTING ==============================
    fig = plot_bpsk_ber(EbN0_dB, BER_theory, BER_sim, Nbits);
    
    % --- Save figure ---
    save_figure_png(fig, ...
        'Q3_Uncoded_BPSK_AWGN', ...
        'results/Q3');
end

%====================== Q3 Helper functions =============================

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
function fig = plot_bpsk_ber(EbN0_dB, BER_theory, BER_sim, Nbits)
% PLOT_BPSK_BER
% Plots theoretical and simulated BER for uncoded BPSK over AWGN
%
% Inputs:
%   EbN0_dB     : Eb/N0 values in dB
%   BER_theory : theoretical BER
%   BER_sim    : simulated BER
%   Nbits      : number of simulated bits
%
% Output:
%   fig        : figure handle

    fig = figure;

    % --- Theoretical BER ---
    semilogy(EbN0_dB, BER_theory, ...
        'b-', 'LineWidth', 2);
    hold on;

    % --- Simulated BER ---
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

    % ================= TEXT BOX ON FIGURE =================
    infoStr = { ...
        'Modulation: BPSK', ...
        'Channel: AWGN', ...
        sprintf('Monte-Carlo bits: %d', Nbits) ...
    };

    annotation(fig, 'textbox', ...
        [0.15 0.18 0.3 0.15], ... % position [x y w h]
        'String', infoStr, ...
        'FitBoxToText','on', ...
        'BackgroundColor','white', ...
        'EdgeColor','black', ...
        'FontSize',10);
end


%% Q4
% ===============================================================
% ====================== QUESTION 4 =============================
% ===============================================================
function Q4
% BPSK with Repetition-3 Coding (Hard Decision)

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

    %% ===== Case 1: Same Energy per Transmitted Bit =====
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

    %% ===== Case 2: Same Energy per Information Bit =====
    A_info = A / sqrt(R);
    tx_symbols = A_info * (2*coded_bits - 1);

    for k = 1:length(EbN0_dB)
        sigma = sqrt((Eb/2) / EbN0_lin(k));
        noise = sigma * randn(length(tx_symbols),1);
        rx = tx_symbols + noise;

        % Hard decision
        rx_bits = rx > 0;

        % Majority voting
        rx_matrix = reshape(rx_bits, R, []);
        decoded_bits = sum(rx_matrix,1) >= 2;

        BER_same_Einfo(k) = mean(decoded_bits.' ~= bits);
    end

    %% ===================== PLOTTING ==============================
    fig = plot_q4_ber(EbN0_dB, ...
        BER_uncoded, ...
        BER_same_Etx, ...
        BER_same_Einfo, ...
        Nbits);

    %% ===================== SAVE FIGURE ===========================
    save_figure_png(fig, ...
        'Q4_BPSK_Repetition3_HardDecision', ...
        'results/Q4');
end
%====================== Q4 Helper functions =============================
%% Plot Q4
function fig = plot_q4_ber(EbN0_dB, BER_uncoded, BER_Etx, BER_Einfo, Nbits)

    fig = figure;

    semilogy(EbN0_dB, BER_uncoded, 'k-', 'LineWidth', 2); hold on;
    semilogy(EbN0_dB, BER_Etx, 'r o--', 'LineWidth', 1.5);
    semilogy(EbN0_dB, BER_Einfo, 'b s--', 'LineWidth', 1.5);

    grid on; grid minor;

    xlabel('E_b/N_0 (dB)', 'FontSize',12,'FontWeight','bold');
    ylabel('Bit Error Rate (BER)', 'FontSize',12,'FontWeight','bold');

    title('Figure 2: BPSK with Repetition-3 (Hard Decision)', ...
        'FontSize',14,'FontWeight','bold');

    legend( ...
        'Uncoded BPSK (Theoretical)', ...
        'Repetition-3 (Same E per Tx Bit)', ...
        'Repetition-3 (Same E per Info Bit)', ...
        'Location','southwest');

    ylim([1e-5 1]);
    set(gca,'FontSize',11);

    % Info box
    annotation(fig,'textbox', ...
        [0.15 0.18 0.35 0.18], ...
        'String',{ ...
            'Modulation: BPSK', ...
            'Coding: Repetition-3 (Hard Decision)', ...
            sprintf('Monte-Carlo bits: %d', Nbits)}, ...
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
        'Q6_(7,4)_Hamming code', ...
        'results/Q6');

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
        'results/Q7');

%% ================= REPORT ANSWER (7.1.e) ====================
fprintf('(e) Recommendation:\n');
fprintf(' QPSK is preferred.\n');
fprintf(' Same BER as BPSK with double transmission rate.\n');
fprintf(' More spectrally efficient with no BER penalty.\n');
end

% ===============================================================
function Q8
x = randn(1,10000);
histogram(x,50);
grid on;
title('Q8 Placeholder');
end

% ===============================================================
function Q9
figure;
imagesc(rand(10));
colorbar;
title('Q9 Placeholder');
end
