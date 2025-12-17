%% Q4
% ===============================================================
% ====================== QUESTION 4 =============================
% ===============================================================
function Q4
% BPSK with Repetition-3 Coding (Hard Decision)
    fprintf('Q4 Start\n');
    pause(3);


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
    fprintf('Simulating Case1 \n');
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
    fprintf('Simulating Case2 \n');
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
    fprintf('Q4 Plot \n');

    fig = plot_q4_ber(EbN0_dB, ...
        BER_uncoded, ...
        BER_same_Etx, ...
        BER_same_Einfo, ...
        Nbits);

    %% ===================== SAVE FIGURE ===========================
    save_figure_png(fig, ...
        'Q4_BPSK_Repetition3_HardDecision', ...
        'figures');
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
