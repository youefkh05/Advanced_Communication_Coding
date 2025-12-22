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


%% Q3
% ===============================================================
% ====================== QUESTION 3 =============================
% ===============================================================
function Q3
    
    %% ================= PARAMETERS =================
    Nfft = 256;
    Eb = 1;
    EbN0_dB = 0:2:20;
    EbN0_lin = 10.^(EbN0_dB/10);
    No = Eb ./ EbN0_lin;                 % noise spectral density
    
    mods = {'BPSK','QPSK','16QAM'};
    R = 5;                     % repetition factor
    Nsym = 150;                % OFDM symbols per SNR (runtime safe)
    
    % Two Channels
    h_flat = (randn + 1j*randn)/sqrt(2);
    H = (randn(Nfft,1) + 1j*randn(Nfft,1)) / sqrt(2);
    
    %% ================= LOOP OVER MODULATIONS =================
    for m = 1:length(mods)
    
        modType = mods{m};    
    
        %% ===== OFDM FRAME DEFINITION =====
        switch modType
            case 'BPSK'
                rows = 32; cols = 8;   bits_ps = 1;
            case 'QPSK'
                rows = 32; cols = 16;  bits_ps = 2;
            case '16QAM'
                rows = 32; cols = 32;  bits_ps = 4;
        end
        
        bits_per_symbol = rows * cols;   % mapper input size
        
        % ---- Uncoded system ----
        bits_uncoded = bits_per_symbol;  % full OFDM payload
        
        % ---- Repetition-coded system ----
        bits_info_rep = floor(bits_per_symbol / R);  % information bits
        bits_coded    = bits_info_rep * R;            % after repetition
    
    
    
    
        BER_flat_unc = zeros(size(EbN0_dB));
        BER_flat_rep = zeros(size(EbN0_dB));
        BER_freq_unc = zeros(size(EbN0_dB));
        BER_freq_rep = zeros(size(EbN0_dB));
    
        for snr = 1:length(EbN0_dB)
    
            err_fu = 0; err_fc = 0;
            err_su = 0; err_sc = 0;
            bits_unc_cnt = 0; bits_rep_cnt = 0;
    
            for sym = 1:Nsym
    
                %% ========== PART 1: CODING ==========
    
                % ---- Uncoded ----
                info_unc = randi([0 1], bits_uncoded, 1);
                
                % ---- Repetition coded ----
                info_rep  = randi([0 1], bits_info_rep, 1);
                coded_rep = repelem(info_rep, R);
                
                % ---- Padding to OFDM size ----
                info_unc_p  = info_unc;  % already bits_per_symbol
                coded_rep_p = [coded_rep; zeros(bits_per_symbol - length(coded_rep),1)];
                            
                %% ========== PART 2: INTERLEAVER ==========
                info_unc_i  = ofdm_interleave(info_unc_p,  modType);
                coded_rep_i = ofdm_interleave(coded_rep_p, modType);
    
                %% ========== PART 3: MAPPER ==========
                X_unc = ofdm_mapper(info_unc_i, modType, Eb);
                X_rep = ofdm_mapper(coded_rep_i, modType, Eb);
    
    
                %% ========== PART 4a: IFFT ==========
                x_unc = ifft(X_unc);
                x_rep = ifft(X_rep);
                
                %% ========== PART 5a: FLAT FADING ==========
                    
                y_unc_flat = h_flat * x_unc;
                y_rep_flat = h_flat * x_rep;
    
     
                
                %% ========== AWGN ==========
                sigma = sqrt(Eb ./ (2*EbN0_lin(snr)));
    
                noise_unc = sigma * (randn(size(y_unc_flat)) + 1j*randn(size(y_unc_flat)));
                noise_rep = sigma * (randn(size(y_rep_flat)) + 1j*randn(size(y_rep_flat)));
    
                y_unc_flat = y_unc_flat + noise_unc;
                y_rep_flat = y_rep_flat + noise_rep;
    
                %% ========== PART 5b: FREQUENCY SELECTIVE ==========
                
                y_unc_sel = ifft(X_unc .* H);
                y_rep_sel = ifft(X_rep .* H);
                
                %% ========== AWGN ==========
                sigma = sqrt(Eb ./ (2*EbN0_lin(snr)));
    
                noise_unc = sigma * (randn(size(y_unc_sel)) + 1j*randn(size(y_unc_sel)));
                noise_rep = sigma * (randn(size(y_rep_sel)) + 1j*randn(size(y_rep_sel)));
      
                y_unc_sel = y_unc_sel + noise_unc;
                y_rep_sel = y_rep_sel + noise_rep;
    
    
                %% ========== PART 6: RECEIVER (FLAT) ==========
                err_fu = err_fu + ofdm_receiver(y_unc_flat, h_flat, info_unc, modType, false, R);
                err_fc = err_fc + ofdm_receiver(y_rep_flat, h_flat, info_rep, modType, true,  R);
    
                %% ========== PART 6: RECEIVER (SELECTIVE) ==========
                err_su = err_su + ofdm_receiver(y_unc_sel, H, info_unc, modType, false, R);
                err_sc = err_sc + ofdm_receiver(y_rep_sel, H, info_rep, modType, true,  R);
                           
                bits_unc_cnt = bits_unc_cnt + length(info_unc);
                bits_rep_cnt = bits_rep_cnt + length(info_rep);
    
            end
    
            BER_flat_unc(snr) = err_fu / bits_unc_cnt;
            BER_flat_rep(snr) = err_fc / bits_rep_cnt;
            BER_freq_unc(snr) = err_su / bits_unc_cnt;
            BER_freq_rep(snr) = err_sc / bits_rep_cnt;
        end
    
        %% ========== PLOT ==========
        fig = plot_ofdm(EbN0_dB, ...
            BER_flat_unc, BER_flat_rep, ...
            BER_freq_unc, BER_freq_rep, modType);
    
        save_figure_png(fig,['Q3_OFDM_' modType],'figures');
    end

end
%====================== Q3 Helper functions =============================

%% Coding
function [info_unc, info_rep, coded_rep, pad_unc, pad_rep] = ofdm_coding(modType,Nfft,R)

    bps = strcmp(modType,'BPSK') + ...
          2*strcmp(modType,'QPSK') + ...
          4*strcmp(modType,'16QAM');

    Ninfo_unc = floor(Nfft/bps);
    Ninfo_rep = floor(Nfft/(R*bps));

    info_unc = randi([0 1],Ninfo_unc*bps,1);
    info_rep = randi([0 1],Ninfo_rep*bps,1);

    coded_rep = repelem(info_rep,R);

    % ---- padding length (DO NOT append yet) ----
    pad_unc = Nfft*bps - length(info_unc);
    pad_rep = Nfft*bps - length(coded_rep);
end

%% InterLeaver
function out = ofdm_interleave(bits, modType)

    switch modType
        case 'QPSK'
            assert(length(bits)==512, 'QPSK interleaver requires 512 bits');
            out = reshape(bits,32,16).';
            out = out(:);

        case '16QAM'
            assert(length(bits)==1024,'16QAM interleaver requires 1024 bits');
            out = reshape(bits,32,32).';
            out = out(:);

        otherwise  % BPSK
            out = bits;
    end
end

%% Mapper 
function X = ofdm_mapper(bits, modType, Eb)

    switch modType

        case 'BPSK'
            % Average symbol energy = Eb
            X = sqrt(Eb) * (2*bits - 1);

        case 'QPSK'
            % Average symbol energy = 2Eb → scaled to Eb
            b = reshape(bits,2,[]).';
            X = sqrt(Eb) * ( ...
                (2*b(:,1)-1) + 1j*(2*b(:,2)-1) );

        case '16QAM'
            % Average symbol energy = 10 → normalized by 2.5
            b = reshape(bits,4,[]).';
            I = (2*b(:,1)-1).*(2-(2*b(:,3)));
            Q = (2*b(:,2)-1).*(2-(2*b(:,4)));
            X = sqrt(Eb/2.5) * (I + 1j*Q);

    end
end

%% Recevier
function err = ofdm_receiver(y, h, info_bits, modType, isRep, R)

    %% FFT
    Y = fft(y);

    %% Equalization
    if numel(h) > 1        % frequency selective
        Yeq = Y ./ (h + 1e-12);
    else                  % flat fading
        Yeq = Y / h;
    end

    %% Demapper
    switch modType
        case 'BPSK'
            rx_bits = real(Yeq) > 0;

        case 'QPSK'
            rx_bits = reshape([real(Yeq)>0 imag(Yeq)>0].',1,[]);

        case '16QAM'
            rx_bits = reshape([ ...
                real(Yeq)>0 imag(Yeq)>0 ...
                abs(real(Yeq))<2 abs(imag(Yeq))<2].',1,[]);
    end

    %% De-interleaver
    switch modType
        case 'QPSK'
            rx_bits = reshape(rx_bits,16,32).';
            rx_bits = rx_bits(:).';

        case '16QAM'
            rx_bits = reshape(rx_bits,32,32).';
            rx_bits = rx_bits(:).';
    end

    %% Decode (Repetition)
    if isRep
        dec = zeros(1,length(info_bits));
        for k = 1:length(info_bits)
            dec(k) = sum(rx_bits((k-1)*R+1:k*R)) > R/2;
        end
        rx_bits = dec;
    else
        rx_bits = rx_bits(1:length(info_bits));
    end

    %% Error count
    rx_bits  = rx_bits(:);
    info_bits = info_bits(:);
    
    err = sum(rx_bits ~= info_bits);

end

%% Noise
function n = awgn_noise(x, Eb, EbN0)
    N0 = Eb / EbN0;
    n  = sqrt(N0/2) * (randn(size(x)) + 1j*randn(size(x)));
end


%% Plot
function fig = plot_ofdm(EbN0_dB,fU,fC,sU,sC,modType)
    fig = figure;
    semilogy(EbN0_dB,fU,'r-o','LineWidth',1.6); hold on;
    semilogy(EbN0_dB,fC,'b-s','LineWidth',1.6);
    semilogy(EbN0_dB,sU,'k-^','LineWidth',1.6);
    semilogy(EbN0_dB,sC,'m-d','LineWidth',1.6);
    grid on; grid minor;
    legend('Flat-NoCode','Flat-Rep','Freq-NoCode','Freq-Rep','Location','southwest');
    xlabel('E_b/N_0 (dB)');
    ylabel('BER');
    title(['OFDM ' modType ' over Flat & Frequency Selective Fading']);
end
