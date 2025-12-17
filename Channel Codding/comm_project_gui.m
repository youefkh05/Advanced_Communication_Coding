function comm_project_gui
% DIGITAL COMMUNICATIONS PROJECT GUI (Q3 → Q9)
% Single-file implementation

    close all;

    % ================= HOME PAGE =================
    homeFig = uifigure( ...
        'Name','Digital Communications Project', ...
        'Position',[500 200 420 500]);

    gl = uigridlayout(homeFig,[8 1]);
    gl.RowHeight = {'fit','1x','1x','1x','1x','1x','1x','1x'};
    gl.Padding = [20 20 20 20];

    % Title
    uilabel(gl, ...
        'Text','Digital Communications Project', ...
        'FontSize',18, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','center');

    % Question buttons
    for q = 3:9
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

    homeFig.Visible = 'off';

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

    % Status
    status = uilabel(gl, ...
        'Text','Running code...', ...
        'FontSize',12, ...
        'HorizontalAlignment','center');

    drawnow;

    % Run selected question
    try
        switch qnum
            case 3
                Q3;
            case 4
                Q4;
            case 5
                Q5;
            case 6
                Q6;
            case 7
                Q7;
            case 8
                Q8;
            case 9
                Q9;
        end
        status.Text = 'Execution completed successfully.';
    catch ME
        status.Text = 'Error occurred!';
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

% ===============================================================
% ====================== QUESTION 3 =============================
% ===============================================================
function Q3
% Uncoded BPSK over AWGN

EbN0_dB = -3:1:10;
EbN0_lin = 10.^(EbN0_dB/10);
Eb = 1;
A = sqrt(Eb);
Nbits = 110000;

BER_theory = 0.5 * erfc(sqrt(EbN0_lin));

tx_bits = randi([0 1], Nbits, 1);
tx_symbols = A * (2*tx_bits - 1);

BER_sim = zeros(size(EbN0_dB));

for k = 1:length(EbN0_dB)
    sigma = sqrt((Eb/2) / EbN0_lin(k));
    noise = sigma * randn(Nbits,1);
    rx = tx_symbols + noise;
    rx_bits = rx > 0;
    BER_sim(k) = mean(rx_bits ~= tx_bits);
end

figure('Name','Q3 Result');
semilogy(EbN0_dB, BER_theory,'k','LineWidth',2); hold on;
semilogy(EbN0_dB, BER_sim,'ko','LineWidth',1.5);
grid on;
xlabel('E_b/N_0 (dB)');
ylabel('BER');
legend('Theoretical','Simulation');
title('Q3: Uncoded BPSK over AWGN');
end

% ===============================================================
% ====================== QUESTION 4 =============================
% ===============================================================
function Q4
figure;
plot(randn(100,1),'LineWidth',1.5);
grid on;
title('Q4 Placeholder');
end

% ===============================================================
function Q5
figure;
stem(randi([0 1],20),'filled');
grid on;
title('Q5 Placeholder');
end

% ===============================================================
function Q6
t = 0:0.001:1;
plot(t,sin(2*pi*10*t),'LineWidth',2);
grid on;
title('Q6 Placeholder');
end

% ===============================================================
function Q7
EbN0 = 0:10;
plot(EbN0,exp(-EbN0/3),'LineWidth',2);
grid on;
title('Q7 Placeholder');
end

% ===============================================================
function Q8
x = randn(1,10000);
histogram(x,50);
grid on;
title('Q8 Placeholder');
end

% ===============================================================
function Q9
figure;
imagesc(rand(10));
colorbar;
title('Q9 Placeholder');
end
