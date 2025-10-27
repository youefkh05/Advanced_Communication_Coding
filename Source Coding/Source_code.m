%%------------------------------------------------------------ 
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

% Print the dictionary neatly
print_symbols_dic(dict_input, H);


% -------------------------------------------------------------------------
% Manual Huffman Coding (with custom output)
% -------------------------------------------------------------------------
dict_huffman = huffman_encoding_visual(dict_input);

disp('--- Manual Huffman Encoding ---');
disp(dict_huffman);

% Print the codded dictionary neatly
print_coded_dict(dict_huffman, H);


%%------------------------------------------------------------ 
% Problem 2: Binary Fano  Coding 
%------------------------------------------------------------ 
 
%{
الجزء بتاعكوا يا رجالة
استخدموا ده لل
input 
بتاعكوا
dict_input

ال
funtions 
H
L
eta
دول جاهزين ليكوا

اللي عليكوا تعملوه  هو 
dict_Fano  = Fano_encoding_visual(dict_input);
اللي هي ال function 
اللي تخت خالص
%}

% -------------------------------------------------------------------------
% Manual Fano Coding (with custom output)
% -------------------------------------------------------------------------
dict_Fano  = Fano_encoding_visual(dict_input);

disp('--- Manual Fano Encoding ---');
disp(dict_Fano);

% Print the codded dictionary neatly
print_coded_dict(dict_Fano, H);



%%  
% -------------------------------------------------------------------------
%              Function Definition
% -------------------------------------------------------------------------

%% -------------------------------------------------------------------------
%              Entropy Calculation 
% -------------------------------------------------------------------------

function H = entropy_calc(P)
%ENTROPY_CALC Compute the source entropy H(P(x))
%   H = entropy_calc(P)
%   P : vector of symbol probabilities
%   H : entropy in bits

    % Validate input
    if any(P < 0) || abs(sum(P) - 1) > 1e-6
        warning('Probabilities should sum to 1. Normalizing...');
        P = P / sum(P);
    end

    % Remove zeros (to avoid log2(0))
    P = P(P > 0);

    % Compute entropy
    H = -sum(P .* log2(P));
end

%% -------------------------------------------------------------------------
%              Average Length Calculation 
% -------------------------------------------------------------------------
function L = average_length_calc(dict)
%AVERAGE_LENGTH_CALC Compute average codeword length L(C)
%   L = average_length_calc(dict, P)
%   dict : Huffman dictionary cell array {symbol, code}
%   P : vector of symbol probabilities (same order as dict)
%
%   If P is empty, it tries to extract from dict(:,2) if present
%   L : average code length

    P = dict(:,2);
    % --- Handle inputs ---
    if nargin < 2 || isempty(P)
        % Check if dict has a probability column (3 columns)
        if size(dict, 2) >= 3 && isnumeric(dict{1,2})
            P = cell2mat(dict(:,2));
            codes = dict(:,3);
        else
            error('Probability vector P is required or must be in dict(:,2)');
        end
    else
        % Extract codes (assumed in 2nd column)
        codes = dict(:,3);
    end

    % --- Compute code lengths ---
    code_lengths = cellfun(@length, codes);

    % --- Normalize probabilities ---
    P = P(:) / sum(P);

    % --- Check dimensions ---
    if length(P) ~= length(code_lengths)
        error('Number of probabilities does not match number of codewords.');
    end

    % --- Compute average code length ---
    L = sum(P .* code_lengths);
end

%% -------------------------------------------------------------------------
%              Efficiency Calculation
% -------------------------------------------------------------------------

function eta = efficiency_calc(H, L)
%EFFICIENCY_CALC Compute Huffman coding efficiency η
%   eta = efficiency_calc(H, L)
%   H : entropy
%   L : average code length
%   eta : efficiency in percentage (%)

    if L <= 0
        error('Average length L must be positive.');
    end

    eta = (H / L) * 100;
end


%% -------------------------------------------------------------------------
%               Print Kraft Inequality Function
% -------------------------------------------------------------------------

function [kraft_sum, kraft_flag]=kraft_analysis(dict)
% KRAFT_ANALYSIS  Compute Kraft's inequality and visualize Kraft tree
%   kraft_analysis(dict)
%
%   Input:
%       dict : cell array {N x 3}  → {symbol, P, code}
%
%   Example:
%       dict = {'A','0'; 'B','10'; 'C','110'; 'D','111'};
%       kraft_analysis(dict);

    if ~iscell(dict) || size(dict,2) < 2
        error('Input must be a cell array {symbol, code}');
    end

    % Extract codes
    codes = dict(:,3);
    N = length(codes);

    % --- 1. Compute Kraft's inequality ---
    code_lengths = cellfun(@length, codes);
    kraft_sum = sum(2.^(-code_lengths));

    fprintf('\n=== Kraft Inequality Check ===\n');
    fprintf('Sum(2^{-l_i}) = %.4f\n', kraft_sum);



    if abs(kraft_sum - 1) < 1e-6
        fprintf('✅ Code satisfies equality → Complete Prefix Code.\n');
        kraft_flag=2;
    elseif kraft_sum < 1
        fprintf('⚠ Code satisfies inequality (valid but not complete).\n');
        kraft_flag=1;
    else
        fprintf('❌ Invalid prefix code (violates Kraft''s inequality).\n');
        kraft_flag=0;
    end

end


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
    H = entropy_calc(P)
    
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
        return;
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
%               Print Coded Dictionary Function (with Kraft Tree)
% -------------------------------------------------------------------------
function print_coded_dict(dict, H)
% PRINT_CODED_DICT  Display Huffman dictionary with entropy, avg length, efficiency, and Kraft tree.
%
%   print_coded_dict(dict, H)
%
%   Inputs:
%       dict - cell array {symbol, probability, code}
%       H    - entropy (bits/symbol)
%
%   This function:
%       • Calculates average length L(C)
%       • Calculates efficiency η = (H / L) * 100%
%       • Checks Kraft’s inequality and plots the Kraft tree
%       • Displays all results in MATLAB UI + console

    % === Validate input ===
    if nargin < 1 || isempty(dict)
        disp('Input Huffman dictionary is missing or empty.');
        return;
    end

    if size(dict,2) < 3
        disp('Dictionary must have 3 columns: {symbol, probability, code}.');
        return;
    end
    
    % === Extract data ===
    symbols = cellfun(@char, dict(:,1), 'UniformOutput', false);
    P       = cell2mat(dict(:,2));
    codes   = dict(:,3);

    % === Compute metrics ===
    L   = average_length_calc(dict);
    eta = efficiency_calc(H, L);
    [kraft_sum, kraft_flag] = kraft_analysis(dict); 

    % === Print to Command Window ===
    fprintf('\n--- Final Huffman Coding Results ---\n');
    fprintf('Symbol\tProb.\t\tCode\n');
    fprintf('-----------------------------------------\n');
    for i = 1:length(symbols)
        fprintf('%s\t%.4f\t\t%s\n', symbols{i}, P(i), codes{i});
    end
    fprintf('-----------------------------------------\n');
    fprintf('Entropy (H):           %.4f bits/symbol\n', H);
    fprintf('Average length (L):    %.4f bits/symbol\n', L);
    fprintf('Efficiency (η):        %.2f %%\n', eta);
    fprintf('Kraft Sum:             %.4f\n', kraft_sum);
    if kraft_flag == 2
        fprintf('Kraft Result: ✅ Complete Prefix Code\n');
    elseif kraft_flag == 1
        fprintf('Kraft Result: ⚠ Valid but Not Complete\n');
    else
        fprintf('Kraft Result: ❌ Invalid Code\n');
    end

    % === UI Figure ===
    f = uifigure('Name','Huffman Dictionary Summary', ...
                 'NumberTitle','off', ...
                 'Color','w', ...
                 'Position',[500 200 480 450]);

    gl = uigridlayout(f,[3 1]);
    gl.RowHeight = {'fit', '1x', 'fit'};
    gl.Padding = [10 10 10 10];

    % --- Title ---
    uilabel(gl, ...
        'Text','--- Huffman Coded Dictionary ---', ...
        'FontSize',14, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','center');

    % --- Table ---
    data = [symbols, arrayfun(@(p) sprintf('%.4f',p), P,'UniformOutput',false), codes];
    uitable(gl, ...
        'Data',data, ...
        'ColumnName',{'Symbol','Probability','Code'}, ...
        'FontSize',12, ...
        'RowStriping','on', ...
        'ColumnWidth',{'1x','1x','1x'});

    % --- Summary Labels ---
    uilabel(gl, ...
        'Text', sprintf('H = %.4f | L = %.4f | η = %.2f %% | Kraft = %.4f', H, L, eta, kraft_sum), ...
        'FontSize',12, ...
        'FontWeight','bold', ...
        'FontColor',[0 0.3 0.7], ...
        'HorizontalAlignment','center');

end


%% ------------------------------------------------------------------------- 
%               Huffman Encoding with Visualization Function  
% ------------------------------------------------------------------------- 

function dict = huffman_encoding_visual(dict_input)
%HUFFMAN_ENCODING_VISUAL Visual Huffman encoding with full table output (UI-based)
%
%   dict = huffman_encoding_visual(symbols, P)
%   - symbols: cell array of symbol names (e.g. {'A','B','C','D','E','F','G'})
%   - P: vector of probabilities (same length as symbols)
%
%   Creates a UI figure showing the probability & code propagation table,
%   and prints the final Huffman dictionary.
    
    % get the info from dictionary 
    symbols = dict_input(:,1);
    P = cell2mat(dict_input(:,2));
    % === Input Validation ===
    if numel(symbols) ~= numel(P)
        error('Symbols and probabilities must have same length.');
    end

    % === Normalize probabilities ===
    P = P(:);
    P = P / sum(P);

    % === Step 1: Generate merging history ===
    history_table = merge_probabilities(P);

    % === Step 2: Assign Huffman codes ===
    history_table_full = assign_coding(history_table);

    % === Step 3: Prepare data for visualization ===
    % Convert numeric NaNs to empty strings for table display
    final_visual_data = cell(size(history_table_full));
    for r = 1:size(history_table_full,1)
        for c = 1:size(history_table_full,2)
            val = history_table_full{r,c};
            if isnumeric(val)
                if isnan(val)
                    final_visual_data{r,c} = '';
                else
                    final_visual_data{r,c} = num2str(val, '%.4f');
                end
            else
                final_visual_data{r,c} = val;
            end
        end
    end

    % Generate column headers (P1, C1, P2, C2, ...)
    numCols = size(history_table_full,2);
    final_visual_headers = cell(1,numCols);
    for c = 1:numCols
        if mod(c,2)==1
            final_visual_headers{c} = sprintf('P%d', ceil(c/2)-1);
        else
            final_visual_headers{c} = sprintf('C%d', ceil(c/2)-1);
        end
    end

    % === Step 4: Build UI Visualization ===
    close all;
    f = uifigure('Name','Huffman Encoding Visualization', ...
                 'Position',[100 100 1000 500]);
    gl = uigridlayout(f,[2 1]);
    gl.RowHeight = {'fit','1x'};

    uilabel(gl, ...
        'Text','Huffman Encoding: Probability and Code Evolution (P/C Steps)', ...
        'FontSize',16, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','center');

    % Column widths (narrow for numeric columns, wider for code columns)
    col_widths = repmat({70}, 1, numCols);
    col_widths(2:2:end) = {100}; % widen code columns

    uitable(gl, ...
        'Data',final_visual_data, ...
        'ColumnName',final_visual_headers, ...
        'RowName',{}, ...
        'FontSize',12, ...
        'ColumnWidth',col_widths, ...
        'RowStriping','on', ...
        'BackgroundColor',[1 1 1; 0.95 0.95 1]);

    % === Step 5: Extract Final Huffman Dictionary ===
    
    % Make a copy
    dict = dict_input;
    
    % Ensure dict has at least 3 columns
    if size(dict,2) < 3
        dict(:,end+1:3) = {[]}; 
    end
    dict(:,3)=history_table_full(:,2);
    
    % === Step 6: Console Output ===
    firstPcol = 1;
    firstCcol = 2;

    probs = cell2mat(history_table_full(:, firstPcol));
    codes = history_table_full(:, firstCcol);
    validIdx = ~isnan(probs);

    symbols = symbols(validIdx);
    codes = codes(validIdx);
    probs = probs(validIdx);
    
    fprintf('\n--- Final Huffman Codes ---\n');
    for i = 1:length(symbols)
        fprintf('Symbol %s (%.4f): %s\n', symbols{i}, probs(i), codes{i});
    end
    fprintf('===============================================================\n\n');
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


%% ------------------------------------------------------------------------- 
%               Fano Encoding with Visualization Function  
% ------------------------------------------------------------------------- 
function dict = Fano_encoding_visual(dict_input)
%{هنا شغلكوا
%} 
    dict = [];
end



