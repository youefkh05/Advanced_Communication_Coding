%------------------------------------------------------------ 
% Problem 1: Binary Huffman Coding 
%------------------------------------------------------------ 

clc; clear; close all;

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


% Compute entropy and efficiency
H = -sum(P .* log2(P));
eff = (H / avglen) * 100;

fprintf('\nEntropy (H) = %.4f bits/symbol\n', H);
fprintf('Average code length (L) = %.4f bits/symbol\n', avglen);
fprintf('Coding Efficiency = %.2f%%\n', eff);



% -------------------------------------------------------------------------
% Manual Huffman Coding (with custom output)
% -------------------------------------------------------------------------
dict_manual = huffman_encoding(symbols, P);

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
%%

