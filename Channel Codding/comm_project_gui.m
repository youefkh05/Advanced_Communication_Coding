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
%====================== Q3 Helper functions =============================
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

% ===============================================================
function Q5
figure;
stem(randi([0 1],20),'filled');
grid on;
title('Q5 Placeholder');
end

% ===============================================================
function Q6
t = 0:0.001:1;
plot(t,sin(2*pi*10*t),'LineWidth',2);
grid on;
title('Q6 Placeholder');
end

% ===============================================================
function Q7
EbN0 = 0:10;
plot(EbN0,exp(-EbN0/3),'LineWidth',2);
grid on;
title('Q7 Placeholder');
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
