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
%               Huffman Encoding with Visualization Function (Final)
% -------------------------------------------------------------------------

function [huffman_codes, huffman_tree] = huffman_encoding_visual(symbols, probabilities)
% HUFFMAN_ENCODING_VISUAL
%   Generates Huffman codes and visualizes the step-by-step process
%   with P/C columns (P0 C0, P1 C1, ...) in a MATLAB UI figure.
%   Correctly shows probability reduction (Pk) at each merge step.

    N = length(probabilities);
    if N < 2
        error('At least two symbols are required.');
    end

    % --- Initialization and Sorting -------------------------------------
    initial_order = (1:N)'; % Original indices
    [~, initial_sort_idx] = sort(probabilities, 'descend');
    sorted_symbols = symbols(initial_sort_idx);
    sorted_probs = probabilities(initial_sort_idx);

    % Estimate max table size
    maxCols = 2 * (N - 1) + 2;
    table_history = cell(N, maxCols);

    % Fill P0 (initial probabilities)
    for i = 1:N
        table_history{i,2} = sprintf('%.4f', sorted_probs(i)); % P0
        table_history{i,3} = ''; % C0
    end

    % Initialize nodes (with original index for tracking)
    nodes = struct('Prob', num2cell(probabilities(:)), ...
                   'SymbolIndex', num2cell(initial_order), ...
                   'Children', {[]});
    nodes_working = nodes;

    % --- Huffman Tree Construction and History Capture -------------------
    for k = 1:N-1
        % Sort ascending to find two smallest
        [~, idx] = sort([nodes_working.Prob], 'ascend');
        nodes_working = nodes_working(idx);

        % Create parent node (merge)
        L = nodes_working(1); % Least probability
        R = nodes_working(2);
        parent_prob = L.Prob + R.Prob;
        
        parent = struct('Prob', parent_prob, ...
                        'SymbolIndex', 0, ... % Internal node
                        'Children', { {L, R} });

        % Replace two nodes with parent
        nodes_working(1:2) = [];
        nodes_working(end+1) = parent;

        % Build code map (SymbolIndex -> Code)
        code_map = containers.Map('KeyType', 'double', 'ValueType', 'char');
        % Build probability map (SymbolIndex -> Current Group Probability)
        prob_map = containers.Map('KeyType', 'double', 'ValueType', 'double');
        
        % 1. Traverse the current set of working nodes to get the state (codes and reduced probs)
        for i = 1:numel(nodes_working)
            % Traverse tree (nodes_working(i)) to assign codes (Ck)
            code_map = assign_current_codes_recursively(nodes_working(i), code_map, '');
            
            % Traverse tree (nodes_working(i)) to find group probability (Pk)
            prob_map = get_current_prob_for_symbol(nodes_working(i), prob_map);
        end
        
        % 2. Fill table history (Pk/Ck)
        col_P = 2*k + 2;
        col_C = 2*k + 3;
        if col_C > size(table_history, 2)
            table_history(:, end+1:col_C) = {''};
        end

        for i = 1:N
            original_idx = initial_sort_idx(i); % Row corresponds to this original symbol index
            
            % Pk: Group Probability
            if isKey(prob_map, original_idx)
                current_prob = prob_map(original_idx);
                if current_prob > 0
                    table_history{i, col_P} = sprintf('%.4f', current_prob);
                end
            end
            
            % Ck: Current Code
            if isKey(code_map, original_idx)
                table_history{i, col_C} = code_map(original_idx);
            end
        end
    end

    % --- Final Code Assignment ------------------------------------------
    huffman_tree = nodes_working;
    code_list = assign_final_codes(huffman_tree, '');
    huffman_codes = cell(N,1);
    for i = 1:size(code_list,1)
        huffman_codes{code_list{i,1}} = code_list{i,2};
    end

    % --- Build Final Visualization Table --------------------------------
    final_visual_headers = {'Symbol'};
    for i = 0:N-2
        final_visual_headers = [final_visual_headers, {['P' num2str(i)]}, {['C' num2str(i)]}];
    end
    final_visual_headers = [final_visual_headers, {'Final Code'}];

    numColsFinal = length(final_visual_headers);
    final_visual_data = cell(N, numColsFinal);
    final_codes_sorted = huffman_codes(initial_sort_idx);

    for i = 1:N
        final_visual_data{i,1} = sorted_symbols{i};
        final_visual_data{i,end} = final_codes_sorted{i};
    end

    colsToCopy = min(size(final_visual_data,2)-2, size(table_history,2)-1);
    final_visual_data(:,2:1+colsToCopy) = table_history(:,2:1+colsToCopy);

    % Clean up zeros
    for r = 1:N
        for c = 2:size(final_visual_data,2)-1
            if ischar(final_visual_data{r,c}) && ...
               (strcmp(final_visual_data{r,c},'0.0000') || strcmp(final_visual_data{r,c},''))
                final_visual_data{r,c} = '';
            end
        end
    end

    % --- Visualization UI ----------------------------------------------
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

    col_widths = repmat({70}, 1, size(final_visual_data,2));
    col_widths{1} = 80;
    col_widths{end} = 120;

    uitable(gl, ...
        'Data',final_visual_data, ...
        'ColumnName',final_visual_headers, ...
        'RowName',{}, ...
        'FontSize',12, ...
        'ColumnWidth',col_widths, ...
        'RowStriping','on', ...
        'BackgroundColor',[1 1 1; 0.95 0.95 1]);

    % --- Console Output -------------------------------------------------
    fprintf('\n--- Final Huffman Codes ---\n');
    for i = 1:N
        fprintf('Symbol %s (%.4f): %s\n', symbols{i}, probabilities(i), huffman_codes{i});
    end
end


%% -------------------------------------------------------------------------
%                        Helper Functions
% -------------------------------------------------------------------------

function map = assign_current_codes_recursively(node, map, prefix)
% Generates the code prefix (Ck) for all leaf nodes under the current node.
    if node.SymbolIndex ~= 0
        % Leaf node (original symbol)
        map(node.SymbolIndex) = prefix;
    elseif ~isempty(node.Children)
        % Internal node: traverse left (assign '1') and right (assign '0')
        % Note: The convention here (L=1, R=0) is for temporary table filling
        map = assign_current_codes_recursively(node.Children{1}, map, [prefix '1']);
        map = assign_current_codes_recursively(node.Children{2}, map, [prefix '0']);
    end
end

function prob_map = get_current_prob_for_symbol(node, prob_map)
% Finds the group probability (Pk) for all symbols under the current node.
% The node.Prob is the reduced/combined probability for the group.
    if node.SymbolIndex ~= 0
        % Leaf node: its current "group" probability is its own original probability
        prob_map(node.SymbolIndex) = node.Prob;
    elseif ~isempty(node.Children)
        % Internal node: its probability is the combined probability (Pk)
        
        % 1. Find all original symbols (leaves) under this merged node
        symbol_indices = get_descendant_indices(node);
        
        % 2. Assign the internal node's probability (node.Prob) to all its leaves
        for idx = 1:length(symbol_indices)
            sym_idx = symbol_indices(idx);
            prob_map(sym_idx) = node.Prob;
        end
        
        % 3. Continue traversing to handle any unmerged leaf nodes
        prob_map = get_current_prob_for_symbol(node.Children{1}, prob_map);
        prob_map = get_current_prob_for_symbol(node.Children{2}, prob_map);
    end
end

function indices = get_descendant_indices(node)
% Returns all leaf indices under a given node
    if node.SymbolIndex ~= 0
        indices = node.SymbolIndex;
        return;
    end
    if isempty(node.Children)
        indices = [];
        return;
    end
    indices = [get_descendant_indices(node.Children{1}), ...
               get_descendant_indices(node.Children{2})];
end

function code_list = assign_final_codes(node, code)
% Generates final Huffman codes from the tree
    if node.SymbolIndex ~= 0
        code_list = {node.SymbolIndex, code};
        return;
    end
    
    % The merge step keeps track of L (smallest) and R (second smallest),
    % but the final tree structure means Children{1} is L and Children{2} is R.
    % We must assign codes based on the children's probabilities at this final stage.
    
    % Convention: Smaller probability group gets the longer code ('1' prefix), larger gets '0'.
    if node.Children{1}.Prob <= node.Children{2}.Prob
        % Child 1 (L) is smaller/equal, gets '1'
        left = assign_final_codes(node.Children{1}, [code '1']);
        right = assign_final_codes(node.Children{2}, [code '0']);
    else
        % Child 1 (L) is larger, gets '0'
        left = assign_final_codes(node.Children{1}, [code '0']);
        right = assign_final_codes(node.Children{2}, [code '1']);
    end
    code_list = [left; right];
end
