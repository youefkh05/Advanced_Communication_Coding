
%% Q8
% ===============================================================
% ====================== QUESTION 8 =============================
% ===============================================================
function Q8
% QPSK vs 16-QAM with BCH(255,131)

fprintf('Q8: QPSK vs 16-QAM + BCH(255,131)\n');
fprintf('---------------------------------\n');
pause(3);

%% ================= PARAMETERS ===============================
EbN0_dB = 5:15;
EbN0_lin = 10.^(EbN0_dB/10);

Eb_QPSK  = 1;     % QPSK Eb
Eb_16QAM = 2.5;   % Given in problem

n = 255; 
k = 131;
R = k/n;

MAX_ERR = 300;
CHUNK = 26200;

BER_QPSK   = nan(size(EbN0_dB));
BER_16QAM  = nan(size(EbN0_dB));

%% ================= MAIN LOOP ===============================
for ii = 1:length(EbN0_dB)

    % --- Bit limits per SNR ---
    if EbN0_dB(ii) <= 8
        MAX_BITS = 2.62e6;
    elseif EbN0_dB(ii) == 9
        MAX_BITS = 2.62e7;
    else
        continue   % interpolation later
    end

    %% ================= QPSK ===============================
    err = 0; bits_cnt = 0;
    while err < MAX_ERR && bits_cnt < MAX_BITS
        bits = randi([0 1],1,CHUNK);

        tx = modQPSK(bits);
        sigma = sqrt(Eb_QPSK/(2*10^(EbN0_dB(ii)/10)));
        rx = tx + sigma*(randn(size(tx))+1j*randn(size(tx)));

        bits_hat = demodQPSK(rx);
        err = err + sum(bits ~= bits_hat);
        bits_cnt = bits_cnt + CHUNK;
    end
    BER_QPSK(ii) = err / bits_cnt;

    %% ================= 16-QAM + BCH =========================
    err = 0; bits_cnt = 0;
    while err < MAX_ERR && bits_cnt < MAX_BITS

        bits = randi([0 1],1,CHUNK);
        bits = bits(1:floor(length(bits)/k)*k);

        msg = reshape(bits,k,[])';
        coded = bchenc(gf(msg),n,k);
        codedBits = reshape(double(coded.x)',1,[]);

        pad = mod(4-mod(length(codedBits),4),4);
        codedBits = [codedBits zeros(1,pad)];

        tx = mod16(codedBits);
        Eb_info = Eb_16QAM/R;
        sigma = sqrt(Eb_info/(2*10^(EbN0_dB(ii)/10)));
        rx = tx + sigma*(randn(size(tx))+1j*randn(size(tx)));

        coded_hat = demod16(rx);
        coded_hat = coded_hat(1:numel(codedBits));

        rxMat = reshape(coded_hat,n,[])';
        decoded = bchdec(gf(rxMat),n,k);
        bits_hat = reshape(double(decoded.x)',1,[]);

        err = err + sum(bits ~= bits_hat);
        bits_cnt = bits_cnt + length(bits);
    end
    BER_16QAM(ii) = err / bits_cnt;

    fprintf('Eb/N0=%d dB | QPSK=%.3e | 16QAM+BCH=%.3e\n', ...
        EbN0_dB(ii), BER_QPSK(ii), BER_16QAM(ii));
end

%% ================= INTERPOLATION ============================
BER_QPSK(6:end)  = interp1(5:9,BER_QPSK(1:5),10:15,'linear');
BER_16QAM(6:end) = interp1(5:9,BER_16QAM(1:5),10:15,'linear');

%% ================= PLOTTING ================================
fig = figure; hold on;

semilogy(EbN0_dB,BER_QPSK,'bo--', 'LineWidth', 1.5, ...
    'MarkerSize', 7, 'DisplayName','QPSK (No Coding)');

semilogy(EbN0_dB,BER_16QAM,'go--', 'LineWidth', 1.5, ...
    'MarkerSize', 7, 'DisplayName','16-QAM + BCH(255,131)');

grid on; grid minor;
xlabel('E_b/N_0 (dB)','FontWeight','bold');
ylabel('BER','FontWeight','bold');
title('BER Comparison: QPSK vs 16-QAM + BCH');
legend('Location','southwest');
ylim([1e-6 1]);
set(gca,'YScale','log');

annotation(fig,'textbox', ...
    [0.15 0.18 0.45 0.18], ...
    'String',{ ...
        'Eb = 2.5', ...
        'BCH(255,131)', ...
        'QPSK: higher rate, no coding', ...
        '16-QAM+BCH: higher coding gain'}, ...
    'FitBoxToText','on', ...
    'BackgroundColor','white');

save_figure_png(fig,'Q8_QPSK_vs_16QAM_BCH','figures');
end

%====================== Q8 Helper functions =============================
%%  QPSK Modulator
function sym = modQPSK(bits)
% MODQPSK - QPSK Modulator (Gray mapping)
% Input : bits (row vector, length multiple of 2)
% Output: complex QPSK symbols

    bits = reshape(bits, 2, []).';
    const = [1+1j, -1+1j, -1-1j, 1-1j]; % Gray mapping
    sym = const(bi2de(bits,'left-msb')+1).';
end

%%  QPSK Demodulator
function bits = demodQPSK(sym)
% DEMODQPSK - Hard-decision QPSK demodulator

    const = [1+1j, -1+1j, -1-1j, 1-1j];
    bits = zeros(1, 2*length(sym));

    for k = 1:length(sym)
        [~, idx] = min(abs(sym(k) - const));
        bits(2*k-1:2*k) = de2bi(idx-1, 2, 'left-msb');
    end
end

%%  16 QAM Modulator
function rxsig = mod16(txbits)
    psk16mod = [ ...
         1+1j  3+1j  1+3j  3+3j ...
         1-1j  3-1j  1-3j  3-3j ...
        -1+1j -3+1j -1+3j -3+3j ...
        -1-1j -3-1j -1-3j -3-3j ];

    m = 4;
    sigqam16 = reshape(txbits,m,[])';
    rxsig = psk16mod(bi2de(sigqam16,'left-msb')+1);
end

%%  16 QAM Demodulator
function rxbits = demod16(rxsig)
    m = 4;
    psk16demod = [15 14 6 7 13 12 4 5 9 8 0 1 11 10 2 3];

    rxsig(real(rxsig)>3)  = 3 + 1j*imag(rxsig(real(rxsig)>3));
    rxsig(imag(rxsig)>3)  = real(rxsig(imag(rxsig)>3)) + 1j*3;
    rxsig(real(rxsig)<-3) = -3 + 1j*imag(rxsig(real(rxsig)<-3));
    rxsig(imag(rxsig)<-3) = real(rxsig(imag(rxsig)<-3)) - 1j*3;

    rxdemod = round(real((rxsig+3+1j*3)/2)) + ...
              1j*round(imag((rxsig+3+1j*3)/2));

    rxdebi = real(rxdemod) + 4*imag(rxdemod);
    sigbits = de2bi(psk16demod(rxdebi+1),m,'left-msb');
    rxbits = reshape(sigbits.',1,[]);
end
