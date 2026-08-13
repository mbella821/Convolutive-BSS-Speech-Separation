function RTF = estimate_rtf(X_tf, Pf, eta, ref_mic, target_mic)
% ESTIMATE_RTF Blind estimation of Relative Transfer Functions.
%   RTF = estimate_rtf(X_tf, Pf, eta, ref_mic, target_mic) estimates the
%   RTFs F^{(p)}_{ij}(f) defined in Eq. (FiltresSep3) of the paper.
%
%   Input:
%     X_tf      - M x T x F observation tensor
%     Pf        - N x T x F probabilistic mask tensor (after permutation alignment)
%     eta       - threshold in [0,1] for selecting dominant TF bins (Eq. Eq_mjk12)
%     ref_mic   - index i of the reference microphone
%     target_mic- index p of the target microphone
%   Output:
%     RTF       - N x F matrix of RTF values F^{(p)}_{ij}(f)
%
%   For each source j and frequency f, the temporal indices n_{jf} are
%   selected as those where M_j(t,f) >= eta * max(M_j(:,f)). The RTF is
%   then computed as the mean ratio X_p(n_{jf},f) / X_i(n_{jf},f).

[~, T, F] = size(X_tf);
N = size(Pf, 1);
RTF = zeros(N, F);

for j = 1:N
    for f = 1:F
        % Mask of source j at frequency f
        M_j = Pf(j, :, f); % 1 x T

        % Maximum mask value at this frequency
        max_M = max(M_j);

        % Select dominant TF bins (Eq. Eq_mjk12)
        idx = find(M_j >= eta * max_M);

        % Estimate RTF as the mean ratio over selected bins
        if ~isempty(idx)
            ratios = X_tf(target_mic, idx, f) ./ X_tf(ref_mic, idx, f);
            % Exclude NaN and Inf values
            ratios = ratios(isfinite(ratios));
            if ~isempty(ratios)
                RTF(j, f) = mean(ratios);
            else
                RTF(j, f) = 0;
            end
        else
            % Fallback: use global mean ratio if no dominant bin found
            ratios = X_tf(target_mic, :, f) ./ X_tf(ref_mic, :, f);
            ratios = ratios(isfinite(ratios));
            if ~isempty(ratios)
                RTF(j, f) = mean(ratios);
            else
                RTF(j, f) = 0;
            end
        end
    end
end

% Safeguard against numerical issues
RTF(isnan(RTF)) = 0;
RTF(isinf(RTF)) = 0;
end
