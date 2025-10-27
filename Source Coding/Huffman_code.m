%------------------------------------------------------------ 
% Problem 1: Binary Huffman Coding 
%------------------------------------------------------------ 

clc; clear; close all force;

% Given Symbols probabilities
symbols = {'A','B','C','D','E','F','G'};
P = [0.35 0.30 0.20 0.10 0.04 0.005 0.005];

% Create Input Dictionary
[dict_input,err_flag, H] = create_symbols_dictionary(symbols, P);

% Check Input
if err_flag ==1
    disp('⚠ Stopping execution due to invalid dictionary.');
    return; % exits the current script or function
end

% Generate built-in Huffman dictionary (for verification)
[dict_builtin, avglen] = huffmandict(symbols, P);

disp('--- Built-in Huffman Codes ---');
for i = 1:length(symbols)
    code = dict_builtin{i,2};
    % Fix nested cell issue (handle {[0 1]} or {0 1} cases)
    if iscell(code)
        code = cell2mat(code);
    end
    fprintf('%s : %s\n', symbols{i}, num2str(code));
end

% Print the dictionary neatly
print_symbols_dic(dict_input, H);
%{
% Compute entropy and efficiency
H = -sum(P .* log2(P));
eff = (H / avglen) * 100;

fprintf('\nEntropy (H) = %.4f bits/symbol\n', H);
fprintf('Average code length (L) = %.4f bits/symbol\n', avglen);
fprintf('Coding Efficiency = %.2f%%\n', eff);
%}


% -------------------------------------------------------------------------
% Manual Huffman Coding (with custom output)
% -------------------------------------------------------------------------
dict_manual = huffman_encoding_visual(symbols, P);

disp('--- Manual Huffman Encoding ---');
disp(dict_manual);

%%  
% -------------------------------------------------------------------------
%              Function Definition
% -------------------------------------------------------------------------


%% -------------------------------------------------------------------------
%              Create Dictionary Input Definition
% -------------------------------------------------------------------------
function [dict_input,err_flag, H] = create_symbols_dictionary(symbols, P)
%CREATE_DICTIONARY  Combines symbols and probabilities into a validated dictionary.
%
%   dict_input = create_dictionary(symbols, P)
%
%   Inputs:
%       symbols - cell array of symbols, e.g. {'A','B','C'}
%       P       - corresponding probabilities (row or column vector)
%
%   Output:
%       dict_input - cell array {symbol, probability}
%
%   Example:
%       symbols = {'A','B','C'};
%       P = [0.5 0.3 0.2];
%       dict_input = create_dictionary(symbols, P);

    % Combine into dictionary-like cell array
    dict_input = [symbols(:), num2cell(P(:))];
    
    % Assume not great until great 
    err_flag = 1;

    % Validate using the check_symbols() function
    [ok, msg] = check_symbols(dict_input);

    % Display validation result
    if ok
        disp('✅ Dictionary is valid!');
        err_flag =0;
    else
        disp(['❌ Error: ' msg]);
        err_flag = 1;
    end

    % ---------------------------------------------------------------------
    % Compute entropy (only if valid)
    % ---------------------------------------------------------------------
    P = cell2mat(dict_input(:, 2)); % extract probabilities
    H = -sum(P .* log2(P));         % Shannon entropy in bits/symbol
    
end


%% -------------------------------------------------------------------------
%               Check Input Validation Function
% -------------------------------------------------------------------------
function [isValid, errMsg] = check_symbols(dict_input)
% CHECK_SYMBOLS  Validates a symbol-probability dictionary
%
%   [isValid, errMsg] = check_symbols(dict_input)
%
%   Input:
%       dict_input : Cell array {N×2}, where first column = symbols,
%                    second column = probabilities
%
%   Output:
%       isValid : Logical true if valid, false otherwise
%       errMsg  : String describing validation error (if any)

    % Default output
    isValid = false;
    errMsg  = '';

    try
        % Extract symbols and probabilities
        symbols = dict_input(:, 1);
        P = cell2mat(dict_input(:, 2));

        % Check same length
        if numel(symbols) ~= numel(P)
            errMsg = 'Symbols and probabilities must have the same length.';
            return;
        end

        % Check probabilities sum to 1 (within tolerance)
        if abs(sum(P) - 1) > 1e-6
            errMsg = sprintf('Probabilities do not sum to 1 (sum = %.6f).', sum(P));
            return;
        end

        % Check all probabilities are positive
        if any(P <= 0)
            errMsg = 'All probabilities must be positive.';
            return;
        end

        % If all checks passed
        isValid = true;

    catch ME
        errMsg = ['Invalid dictionary input: ' ME.message];
    end
end


%% -------------------------------------------------------------------------
%               Print Dictionary Function
% -------------------------------------------------------------------------
function print_symbols_dic(dict_input, H)
% PRINT_SYMBOLS_DIC  Displays a formatted version of the symbol dictionary in a figure,
%                    and shows the calculated source entropy.
%
%   print_symbols_dic(dict_input, H)
%
%   Inputs:
%       dict_input - cell array {symbol, probability}
%       H          - source entropy (bits/symbol)

    % Validate input
    if nargin < 1 || isempty(dict_input)
        error('Input dictionary is empty or missing.');
    end

    % Convert symbols to char (uitable can't handle string objects)
    symbols = cellfun(@char, dict_input(:,1), 'UniformOutput', false);
    probs = cell2mat(dict_input(:,2));

    % Display result in Command Window
    fprintf('\nInformation Source Entropy: H = %.4f bits/symbol\n', H);
    fprintf('-----------------------------------------------------\n');

    % Create a responsive UI figure
    f = uifigure('Name', 'Symbol Dictionary', ...
                 'NumberTitle', 'off', ...
                 'Color', 'w', ...
                 'Position', [500 400 350 320]);

    % Format probabilities as strings
    probStr = arrayfun(@(p) sprintf('%.4f', p), probs, 'UniformOutput', false);

    % Combine into table data
    data = [symbols probStr];

    % Create a grid layout (auto-resizes)
    gl = uigridlayout(f, [3,1]);
    gl.RowHeight = {'fit', '1x', 'fit'};  % title, table, entropy
    gl.ColumnWidth = {'1x'};
    gl.Padding = [10 10 10 10];

    % --- Title ---
    uilabel(gl, ...
        'Text', '--- Input Symbol Dictionary ---', ...
        'FontSize', 14, ...
        'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');

    % --- Table ---
    uitable(gl, ...
        'Data', data, ...
        'ColumnName', {'Symbol', 'Probability'}, ...
        'FontSize', 12, ...
        'ColumnWidth', {'1x', '1x'}, ...
        'RowStriping', 'on');

    % --- Entropy Display ---
    uilabel(gl, ...
        'Text', sprintf('Entropy: H = %.4f bits/symbol', H), ...
        'FontSize', 12, ...
        'FontWeight', 'bold', ...
        'FontColor', [0 0.3 0.7], ...
        'HorizontalAlignment', 'center');
end


%% -------------------------------------------------------------------------
%               Huffman Encoding Function
% -------------------------------------------------------------------------
function dict_out = huffman_encoding(symbols, probabilities)
    
    N = length(probabilities);
    
    % Create cell array for nodes (avoid struct array conflict)
    nodes = cell(N,1);
    for i = 1:N
        nodes{i} = struct('Prob', probabilities(i), ...
                          'Symbol', symbols{i}, ...
                          'Children', []);
    end

    % --- Build Huffman Tree ---
    while numel(nodes) > 1
        % Sort by ascending probability
        [~, idx] = sort(cellfun(@(x)x.Prob, nodes));
        nodes = nodes(idx);

        % Take two smallest
        left = nodes{1};
        right = nodes{2};

        % Merge into new node
        parent = struct('Prob', left.Prob + right.Prob, ...
                        'Symbol', '', ...
                        'Children', { {left, right} });

        % Remove and reinsert
        nodes(1:2) = [];
        nodes{end+1} = parent;
    end

    % --- Assign Huffman Codes ---
    root = nodes{1};
    code_map = containers.Map;

    function assign_codes(node, code)
        if isempty(node.Children)
            code_map(node.Symbol) = code;
        else
            assign_codes(node.Children{1}, [code '0']); % left child → 0
            assign_codes(node.Children{2}, [code '1']); % right child → 1
        end
    end

    assign_codes(root, '');

    % --- Format Output ---
    dict_out = cell(N, 3);
    for i = 1:N
        dict_out{i,1} = symbols{i};
        dict_out{i,2} = probabilities(i);
        dict_out{i,3} = code_map(symbols{i});
    end

    % Display tuple-style
    fprintf('\n--- Output as Tuples ---\n');
    for i = 1:N
        fprintf("('%s', %.4f, '%s')\n", dict_out{i,1}, dict_out{i,2}, dict_out{i,3});
    end
end


%% -------------------------------------------------------------------------
%               Huffman Encoding with Visualization Function 
% -------------------------------------------------------------------------
%% ------------------------------------------------------------------------- 
%               Huffman Encoding with Visualization Function  
% ------------------------------------------------------------------------- 
function dict = huffman_encoding_visual(symbols, P)
%HUFFMAN_ENCODING_VISUAL Visual Huffman encoding with full table output
%
%   dict = huffman_encoding_visual(symbols, P)
%   - symbols: cell array of symbol names (e.g. {'A','B','C','D','E','F','G'})
%   - P: vector of probabilities (same length as symbols)
%
%   Displays the probability merging table step by step, and returns the
%   final dictionary of Huffman codes.

    % === Input Validation ===
    if numel(symbols) ~= numel(P)
        error('Symbols and probabilities must have same length.');
    end

    % === Normalize probabilities ===
    P = P(:);
    P = P / sum(P);

    % === Step 1: Generate merging history ===
    history_table = merge_probabilities(P);

    % === Step 2: Assign Huffman codes through backward propagation ===
    history_table_full = assign_coding(history_table);

    % === Step 3: Visualization ===
    fprintf('\n================ Huffman Encoding Visualization ================\n');
    fprintf('--- Probability Merge History ---\n');
    disp(history_table);

    fprintf('\n--- Probability & Code Propagation Table ---\n');
    disp(history_table_full);

    % === Step 4: Extract Final Codes for Each Symbol ===
    % The first probability column and its corresponding code column
    firstPcol = 1;
    firstCcol = 2;

    % Get codes for non-NaN probabilities
    codes = history_table_full(:, firstCcol);
    probs = cell2mat(history_table_full(:, firstPcol));

    validIdx = ~isnan(probs);
    codes = codes(validIdx);
    symbols = symbols(validIdx);

    % Ensure everything is in same order
    dict = containers.Map(symbols, codes);

    % === Step 5: Display final dictionary ===
    fprintf('\n--- Final Huffman Dictionary ---\n');
    for i = 1:length(symbols)
        fprintf('  %-5s : %s\n', symbols{i}, codes{i});
    end
    fprintf('=================================================================\n\n');
end

% === Probability Merge helper function  ===
function history_table = merge_probabilities(P)
%MERGE_PROBABILITIES Builds Huffman probability merging history (descending)
%
%   history_table = merge_probabilities(P)
%
%   Input:
%       P - vector of symbol probabilities
%
%   Output:
%       history_table - table of probabilities after each merge
%                       Columns: P0, P1, P2, ... (N-1 total)
%
%   Note: Probabilities are shown in descending order.

    % --- Input check ---
    if numel(P) < 2
        error('At least two probabilities are required.');
    end

    % --- Initialization ---
    P = P(:);
    P = sort(P, 'descend'); % sort descending
    N = numel(P);

    % Number of P columns = N - 1
    numCols = N - 1;
    maxRows = N;

    % Initialize history as cell
    history = cell(maxRows, numCols);

    % --- Step 0: Fill P0 (descending order) ---
    for i = 1:maxRows
        history{i,1} = P(i);
    end

    curP = P;

    % --- Iteratively merge ---
    for step = 2:numCols
        % Sort ascending to pick smallest two
        curP = sort(curP, 'ascend');
        if numel(curP) >= 2
            p1 = curP(1);
            p2 = curP(2);
            mergedP = p1 + p2;
            % Remove two smallest and add merged one
            curP = [mergedP; curP(3:end)];
        end
        % Sort descending for display
        curP = sort(curP, 'descend');

        % Fill current column
        for r = 1:maxRows
            if r <= numel(curP)
                history{r,step} = curP(r);
            else
                history{r,step} = NaN;
            end
        end
    end

    % --- Column names ---
    colNames = cell(1,numCols);
    for i = 1:numCols
        colNames{i} = sprintf('P%d', i-1);
    end

    % --- Convert to table ---
    history_table = cell2table(history, 'VariableNames', colNames);
end

% === Assign code helper function  ===
function history_table_full = assign_coding(history_table) 
    % assign_coding - expands the history table and assigns Huffman codes 
    % 
    % Input: 
    %   history_table : numeric matrix or table (probability merging history) 
    % 
    % Output: 
    %   history_table_full : cell array with 2N columns 
    %       Odd columns: probability values 
    %       Even columns: assigned codes 
     
    % If table, convert to numeric array 
    if istable(history_table) 
        history_table = table2array(history_table); 
    end 
     
    % Determine size 
    [numRows, numCols] = size(history_table); 
    newCols = 2 * numCols; 
     
    % Initialize 
    history_table_full = cell(numRows, newCols); 
     
    % === Fill odd columns with probabilities === 
    for col = 1:numCols 
        history_table_full(:, 2 * col - 1) = num2cell(history_table(:, col)); 
    end 
     
    % === Initialize last code column (start with last merge) === 
    lastPcol = 2 * numCols - 1; 
    lastCcol = lastPcol + 1; 
    history_table_full{1, lastCcol} = '0';   
    history_table_full{2, lastCcol} = '1';  
    raw_counter=1;  %for parent assignment 
     
    % === Backward propagation of codes === 
    for col = numCols:-1:2  % start from last column going backward 
        currPcol = 2 * col - 1; 
        currCcol = currPcol + 1; 
        prevPcol = 2 * (col - 1) - 1; 
        prevCcol = prevPcol + 1; 
        raw_counter = raw_counter+1; 
         
        % Get non-NaN values from P prev column 
        prevPvals = cell2mat(history_table_full(:, prevPcol)); 
        prevPvals = prevPvals(~isnan(prevPvals)); 
         
        % Get non-NaN values from C curr column 
        currCvals  = history_table_full(:, currCcol); 
       
         
        % Identify merged value 
        if length(prevPvals) >= 2 
            mergedVal = prevPvals(end) + prevPvals(end-1); 
        else 
            continue; 
        end 
         
        % Find which row in current P col matches the mergedVal 
        currPvals = cell2mat(history_table_full(:, currPcol)); 
        matchIdx = find(abs(currPvals - mergedVal) < 1e-12); 
        if numel(matchIdx) > 1 
            matchIdx = matchIdx(1); % take top one if duplicate 
        end 
         
        % Get parent code 
        parentCode = history_table_full{matchIdx, currCcol}; 
        if isempty(parentCode) 
            parentCode = ''; 
        end 
         
        % === Assign child codes === 
        % Last two rows in previous P column are merged into this parent 
        history_table_full{raw_counter, prevCcol}   = [parentCode '0']; 
        history_table_full{raw_counter+1, prevCcol} = [parentCode '1']; 
         
        % For each previous non-merged row (in display order top->bottom)
        for ii = 1:(raw_counter-1)
            % Skip the rows that were just merged (raw_counter and raw_counter+1)
            
            % Get the probability value in the previous column for this row
            valPrev = history_table_full{ii, prevPcol};
            if isnan(valPrev)
                continue; % nothing to copy
            end

            % Find matching value in the current column (exclude merged parent)
            currMatches = find(abs(currPvals - valPrev) < 1e-12);

            % Remove the matchIdx (the merged parent) if it appears
            currMatches(currMatches == matchIdx) = [];

            if isempty(currMatches)
                continue; % no corresponding match found
            end

            % If there are duplicates (two identical probabilities)
            if numel(currMatches) > 1
                % take both, and copy their codes to the two rows
                history_table_full{ii,   prevCcol} = currCvals{currMatches(1)};
                if (ii+1) <= numRows
                    history_table_full{ii+1, prevCcol} = currCvals{currMatches(2)};
                end
            else
                % single match — copy code directly
                history_table_full{ii, prevCcol} = currCvals{currMatches(1)};
            end
        end

    end 
end 
%%




