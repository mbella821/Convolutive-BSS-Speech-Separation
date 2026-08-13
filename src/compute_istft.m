function y = compute_istft(stftsrc, window, noverlap, sig_length, nfft)
% COMPUTE_ISTFT Inverse Short-Time Fourier Transform.
%   x = compute_istft(S, window, noverlap, sig_length, nfft) reconstructs
%   a real-valued time-domain signal from its STFT representation using
%   the overlap-add (OLA) method.
%
%   Input:
%     S          - nfft x T complex STFT matrix
%     window     - synthesis window (column vector)
%     noverlap   - number of overlapping samples
%     sig_length - desired output signal length in samples
%     nfft       - FFT length
%   Output:
%     x          - reconstructed signal (sig_length x 1)


    % Round fractional step sizes (e.g., 2560.5) to avoid indexing errors
    noverlap = round(noverlap);
    y = zeros(1, sig_length);

    % Slide and overlap-add back to the time-domain
    for j = 0:noverlap:(noverlap * (size(stftsrc, 2) - 1))
        aux = stftsrc(:, 1 + j / noverlap);
        
        % Reconstruct full Hermitian spectrum symmetry
        aux = [aux; conj(aux(end-1:-1:2))];
        aux = real(ifft(aux, 'symmetric'));
        
        % Overlap-add using synthesis window
        y((j + 1):(j + nfft)) = y((j + 1):(j + nfft)) + (aux .* window)';
    end
end

