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
%               Huffman Encoding with Visualization Function (Final)
% -------------------------------------------------------------------------

function [huffman_codes, huffman_tree] = huffman_encoding_visual(symbols, probabilities)
% HUFFMAN_ENCODING_VISUAL
%   Generates Huffman codes and visualizes the step-by-step process
%   with P/C columns (P0 C0, P1 C1, ...) in a MATLAB UI figure.
%
%   Visualization Logic:
%   1. Pk column (Probability): Probability is carried over for non-merged symbols. 
%      For the two merged groups (L and R), the resulting P_parent is applied 
%      to the symbols descending from the R group (the larger group) to keep it 
%      active. The symbols descending from the L group (the smaller group) are 
%      BLANKED OUT (Active=0) to simulate its removal from the sorted list.
%   2. Ck column (Current Code): ONLY the 0 or 1 bit assigned in the current step is 
%      shown for the symbols/groups that were just merged. All other Ck cells are blank.

    N = length(probabilities);
    if N < 2
        error('At least two symbols are required.');
    end

    % --- Initialization and Sorting -------------------------------------
    initial_order = (1:N)'; % Original indices
    % Sort descending by probability to determine fixed table row order (A to G)
    [~, initial_sort_idx] = sort(probabilities, 'descend');
    sorted_symbols = symbols(initial_sort_idx);
    sorted_probs = probabilities(initial_sort_idx);

    % Estimate max table size
    maxCols = 2 * (N - 1) + 2;
    table_history = cell(N, maxCols);

    % Fill P0 (initial probabilities) - P0 is always the initial sorted probability
    for i = 1:N
        table_history{i,2} = sprintf('%.4f', sorted_probs(i)); % P0
        table_history{i,3} = ''; % C0
    end

    % Initialize nodes (with original index for tracking)
    nodes = struct('Prob', num2cell(probabilities(:)), ...
                   'SymbolIndex', num2cell(initial_order), ...
                   'Children', {[]}, ...
                   'MergeHistory', {[]}); % Store [step, codebit]

    % nodes_working will hold the active list of nodes/groups
    nodes_working = nodes(initial_sort_idx);
    
    % Value = [Probability, Active_Flag (1=Active/Show, 0=Blanked)]
    % Keys are the original symbol indices (1 to N)
    current_prob_state = containers.Map('KeyType', 'double', 'ValueType', 'any');
    for i = 1:N
        % Initialize: All groups are active
        current_prob_state(initial_sort_idx(i)) = [sorted_probs(i), 1]; % [Prob, Active]
    end

    % --- Huffman Tree Construction and History Capture -------------------
    for k = 1:N-1
        % Sort ascending to find two smallest
        [~, idx] = sort([nodes_working.Prob], 'ascend');
        nodes_working = nodes_working(idx);

        % L and R are the two nodes with least probability (L <= R)
        % We will use R to represent the new parent probability in the P column.
        L = nodes_working(1); % Least probability (gets '1' in this implementation)
        R = nodes_working(2); % Second least probability (gets '0' in this implementation)
        
        % Assign codes *at this current step*
        L = assign_current_codes_recursively(L, k, '1'); % Assign bit '1'
        R = assign_current_codes_recursively(R, k, '0'); % Assign bit '0')

        % Create parent node (merge)
        parent_prob = L.Prob + R.Prob;
        
        parent = struct('Prob', parent_prob, ...
                        'SymbolIndex', 0, ... % Internal node
                        'Children', { {L, R} }, ...
                        'MergeHistory', {[]});

        % Identify the symbols involved in this merge
        symbols_L = get_descendant_indices(L);
        symbols_R = get_descendant_indices(R);
        symbols_merged_in_this_step = [symbols_L, symbols_R];

        % 1. Update the current probability state map and determine blanking:
        
        % The symbols in the L group (smaller prob) will be BLANKED in Pk (Active=0).
        for idx = symbols_L
            % Mark L group as inactive (blanked in Pk)
            % Keep the current group probability for debugging, but set Active=0
            current_prob_state(idx) = [L.Prob, 0]; % [Prob, Inactive]
        end
        
        % The symbols in the R group (larger prob) will be set to the Parent_Prob (Active=1).
        for idx = symbols_R
            % R group becomes the new group value and remains active
            current_prob_state(idx) = [parent_prob, 1]; % [Parent_Prob, Active]
        end
        
        % 2. Carry over non-merged group values
        % Any other symbol/group not in L or R should carry over its previous active state
        
        % We must check all symbols that were active in P(k-1) and ensure they 
        % remain active and carry their probability *unless* they were just
        % marked inactive (symbols_L). This is implicitly handled by not 
        % iterating over the whole map, but let's make it explicit for correctness.
        
        all_symbol_indices = cell2mat(current_prob_state.keys);
        
        for idx = all_symbol_indices
            % Check if this symbol was part of the previous merge (L or R)
            is_newly_merged = any(idx == symbols_L) || any(idx == symbols_R);
            
            if ~is_newly_merged
                % If it was NOT part of this merge, carry over its state from P(k-1) 
                % but update its value from the previous step, which is already stored
                % in the map. No change is needed here, as the map state is only 
                % modified for L and R above.
                
                % We need to make sure the probability value is set correctly for
                % symbols not involved in the current merge. Since we don't know the
                % *previous* step's probability directly without tracking it, we must
                % iterate through the *nodes_working* list (before removing L and R)
                % to find the current active probability for non-merged groups.
                
                % To simplify, we rely on the rule: If a symbol is not in L or R, 
                % its state remains unchanged from the previous P(k-1) to Pk.
                % The map already holds the latest "active" value for all symbols.
                
                % However, for the very first step P1, symbols A, B, C, D, E are
                % not in L or R. Their state should be [P0, 1]. This is correct 
                % because we only modified L and R.
            end
        end

        % 3. Replace two nodes with parent
        nodes_working(1:2) = [];
        nodes_working(end+1) = parent;

        % 4. Fill table history (Pk/Ck)
        col_P = 2*k + 2;
        col_C = 2*k + 3;
        if col_C > size(table_history, 2)
            table_history(:, end+1:col_C) = {''};
        end
        
        % Store the active groups for Ck (only the symbols getting a new bit)
        codes_assigned = [repmat({'1'}, 1, length(symbols_L)), ...
                          repmat({'0'}, 1, length(symbols_R))];

        for i = 1:N
            original_idx = initial_sort_idx(i); % Row corresponds to this original symbol index
            
            % Pk: Group Probability
            
            % Look up the state for this symbol: [Prob, Active]
            state = current_prob_state(original_idx);
            
            if state(2) == 1 % Active_Flag is 1
                % The group is active (it should show a probability) - show its current probability
                table_history{i, col_P} = sprintf('%.4f', state(1));
            else
                % The group is inactive (it was the 'L' node or has been blanked in a prior step)
                table_history{i, col_P} = '';
            end
            
            % Ck: Current Code (ONLY the single bit assigned in this step)
            idx_in_code_list = find(symbols_merged_in_this_step == original_idx);
            if ~isempty(idx_in_code_list)
                % Symbol is getting a new code bit
                table_history{i, col_C} = codes_assigned{idx_in_code_list(1)};
            else
                % Symbol is NOT getting a new code bit, so Ck is blank
                table_history{i, col_C} = ''; 
            end
        end
        
    end
    
    % --- Final Code Assignment ------------------------------------------
    huffman_tree = nodes_working; % nodes_working now contains the root of the final tree
    code_list = assign_final_codes(huffman_tree(1), '');
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

    % Clean up empty/zero strings
    for r = 1:N
        for c = 2:size(final_visual_data,2)-1
            if ischar(final_visual_data{r,c}) && strcmp(final_visual_data{r,c},'0.0000')
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

function node = assign_current_codes_recursively(node, step_k, codebit)
% Assigns the new code bit ONLY to all leaf nodes under this merged group, 
% and updates the MergeHistory for visualization.
    if node.SymbolIndex ~= 0
        % Leaf node (original symbol) - add new codebit
        node.MergeHistory = [node.MergeHistory; step_k, codebit];
    elseif ~isempty(node.Children)
        % Internal node (group) - pass the new codebit down
        
        % The logic here ensures that the code bit is propagated to all
        % leaf nodes that form this group. This is crucial for the 
        % Ck visualization where all component symbols receive the bit.
        
        % Assign the same code bit to both children recursively
        node.Children{1} = assign_current_codes_recursively(node.Children{1}, step_k, codebit);
        node.Children{2} = assign_current_codes_recursively(node.Children{2}, step_k, codebit);
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
    
    % Convention: Smaller probability group gets '1', larger gets '0'.
    % Children{1} is L (smallest), Children{2} is R (second smallest) from the main loop sort.
    % We ensure consistency:
    if node.Children{1}.Prob <= node.Children{2}.Prob
        % L is smaller/equal, gets '1'
        left = assign_final_codes(node.Children{1}, [code '1']);
        right = assign_final_codes(node.Children{2}, [code '0']);
    else
        % L is larger, gets '0'
        left = assign_final_codes(node.Children{1}, [code '0']);
        right = assign_final_codes(node.Children{2}, [code '1']);
    end
    code_list = [left; right];
end
