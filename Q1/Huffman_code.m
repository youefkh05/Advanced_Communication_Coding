%------------------------------------------------------------
% Cairo University - Faculty of Engineering
% Communications Report (Fall 2025/2026)
% Problem 1: Binary Huffman Coding
%------------------------------------------------------------

clc; clear; close all;

% Given probabilities
P = [0.35 0.30 0.20 0.10 0.04 0.005 0.005];
symbols = {'A','B','C','D','E','F','G'};

% Verify probabilities sum to 1
if abs(sum(P) - 1) > 1e-6
    error('Probabilities do not sum to 1.');
end

% Generate Huffman dictionary
[dict, avglen] = huffmandict(symbols, P);

% Display Huffman codes
disp('--- Huffman Codes ---');
for i = 1:length(symbols)
    fprintf('%s : %s\n', symbols{i}, num2str(cell2mat(dict{i,2})));
end

% Calculate entropy
H = -sum(P .* log2(P));

% Efficiency
eff = (H / avglen) * 100;

% Results
fprintf('\nEntropy (H) = %.4f bits/symbol\n', H);
fprintf('Average code length (L) = %.4f bits/symbol\n', avglen);
fprintf('Coding Efficiency = %.2f%%\n', eff);

% Optional: plot code lengths vs. probabilities
lens = cellfun(@length, dict(:,2));
figure;
stem(P, lens, 'filled');
xlabel('Symbol Probability');
ylabel('Code Length');
title('Huffman Code Length vs Symbol Probability');
grid on;
