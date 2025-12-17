
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

