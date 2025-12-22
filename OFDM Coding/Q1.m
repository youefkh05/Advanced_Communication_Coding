clear;
clc;
close all;
L = 4096;
x = randn(1,L);

tic
X_dft = myDFT(x);
t_dft = toc;

tic
X_fft = fft(x);
t_fft = toc;

disp(['Elapsed time is ', num2str(t_dft), ' seconds.']);
disp(['Elapsed time is ', num2str(t_fft), ' seconds.']);

fig1 = plot_dft_fft_time(t_dft, t_fft);
save_figure_png(fig1, 'Q1_DFT_vs_FFT_Execution_Time', 'figures');


%% ========================= Functions ========================
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
