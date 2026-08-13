function stft = compute_stft(x, window, h, nfft)
% STFT_VM Short-Time Fourier Transform computation.
%
% INPUTS:
%   x       : Input 1D signal vector
%   window  : Weighting analysis window (e.g., hanning)
%   h       : Overlap step size (STFT hop size)
%   nfft    : Number of FFT points for frequency analysis
%
% OUTPUT:
%   stft    : Complex-valued STFT spectrogram matrix

    % Round fractional step sizes (e.g., 2560.5) to avoid indexing errors
    h = round(h);
    
    % Calculate dimensions of target STFT matrix
    rown = ceil((1 + nfft) / 2);             
    coln = 1 + fix((length(x) - length(window)) / h);  
    stft = zeros(rown, coln);               
    
    j = 1;
    % Slide window across signal and compute spectrums
    for i = 1:h:(length(x) - length(window))
        aux = fft(x(i : i + length(window) - 1) .* window, nfft);
        stft(:, j) = aux(1 : rown);
        j = j + 1;
    end
end


