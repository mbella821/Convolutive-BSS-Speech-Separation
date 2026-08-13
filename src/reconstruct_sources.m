function [Y_tf_mic1, Y_tf_mic2] = reconstruct_sources(X_tf, Pf, params)
% RECONSTRUCT_SOURCES RTF-based source recombination (exact original logic).
%   [Y_tf_mic1, Y_tf_mic2] = reconstruct_sources(X_tf, Pf, params)
%   reconstructs the spatial images of N sources on BOTH microphones
%   using the exact logic of the original implementation.
%
%
%   Input:
%     X_tf   - M x T x F observation tensor (M=2 required)
%     Pf     - N x T x F probabilistic mask tensor (after permutation alignment)
%     params - struct with fields:
%                .num_sources      - N
%                .num_mics         - M (must be 2)
%                .mask_threshold   - 1.0 for determined, 0.65 for underdetermined
%   Output:
%     Y_tf_mic1 - N x 1 cell array of F x T STFT matrices (ref mic)
%     Y_tf_mic2 - N x 1 cell array of F x T STFT matrices (target mic)

N = params.num_sources;
M = params.num_mics;

if M ~= 2
    error('reconstruct_sources: currently implemented for M=2 only.');
end

[~, T, F] = size(X_tf);

% Extract observations on both microphones (F x T)
X1 = squeeze(X_tf(1, :, :))'; % mic 1 (reference)
X2 = squeeze(X_tf(2, :, :))'; % mic 2 (target)

%% ========================================================================
% STEP 1: Build masked observations with RAW masks 
%% ========================================================================
Mask_raw = cell(N, 1);
stf_j1 = cell(N, 1); % Mask_j .* X1  (raw, for reconstruction)
stf_j2 = cell(N, 1); % Mask_j .* X2  (raw, for reconstruction)
for j = 1:N
    Mask_raw{j} = squeeze(Pf(j, :, :))'; % F x T
    stf_j1{j} = Mask_raw{j} .* X1;
    stf_j2{j} = Mask_raw{j} .* X2;
end

%% ========================================================================
% STEP 2: ENERGY THRESHOLDING (applied to separate copies for c_j estimation)
%% ========================================================================
Mask = cell(N, 1);  % thresholded copies, used only for c_j estimation
for j = 1:N
    Mask{j} = Mask_raw{j};  % copy
end

for i = 2:F
    Emask = mean(abs(X1(i, :)).^2 + abs(X2(i, :)).^2);
    for j = 1:T
        if (abs(X1(i, j)).^2 + abs(X2(i, j)).^2) < 1 * Emask
            for src = 1:N
                Mask{src}(i, j) = 0;
            end
        end
    end
end

%% ========================================================================
% STEP 3: ESTIMATE c_j(f) using THRESHOLDED masks
%% ========================================================================
c = zeros(N, F);
for j = 1:N
    for i = 2:F
        max_val = max(Mask{j}(i, :));
        aux = find(Mask{j}(i, :) == max_val);
        if numel(aux) == 0
            c(j, i) = c(j, i-1);
        else
            c(j, i) = mean(X2(i, aux) ./ X1(i, aux));
        end
    end
end

%% ========================================================================
% MICROPHONE 1 RECONSTRUCTION (reference microphone)
%% ========================================================================
Y_tf_mic1 = cell(N, 1);
for j = 1:N
    Y_tf_mic1{j} = zeros(F, T);
end

if N == 2
    % ==================== DETERMINED CASE (N=M=2) ====================
    d = zeros(N, F);
    for j = 1:N
        k = setdiff(1:N, j);
        eps_val = c(k, :) - c(j, :);
        eps_val(1) = [];
        eps_val(end) = [];
        if ~isempty(eps_val)
            min_eps = min(abs(eps_val));
        else
            min_eps = 1e-6;
        end
        denom = c(k, :) - c(j, :);
        d(j, :) = 1 ./ (min_eps + denom);
    end
    d(isnan(d)) = 0;
    d(isinf(d)) = 0;

    for j = 1:F
        for i = 1:T
            if Pf(1, i, j) > 1 || Pf(2, i, j) > 1
                Y_tf_mic1{1}(j, i) = stf_j1{1}(j, i);
                Y_tf_mic1{2}(j, i) = stf_j1{2}(j, i);
            else
                Y_tf_mic1{1}(j, i) = X2(j, i) - c(1, j) * X1(j, i);
                Y_tf_mic1{2}(j, i) = X2(j, i) - c(2, j) * X1(j, i);
                Y_tf_mic1{1}(j, i) = d(1, j) * Y_tf_mic1{1}(j, i);
                Y_tf_mic1{2}(j, i) = d(2, j) * Y_tf_mic1{2}(j, i);
            end
        end
    end

else
    % ==================== UNDERDETERMINED CASE (N>M) ====================
    % Pairwise gains: d_{jk} = 1 / (c_k - c_j)
    d = zeros(N, N, F);
    for j = 1:N
        for k = 1:N
            if j == k, continue; end
            denom = c(k, :) - c(j, :);
            d(j, k, :) = 1 ./ denom;
        end
    end
    d(isnan(d)) = 0;
    d(isinf(d)) = 0;

    % Recombine each TF bin
    for j = 1:F
        for i = 1:T
            if Pf(1, i, j) > 0.65 || Pf(2, i, j) > 0.65 || Pf(3, i, j) > 0.65
                for src = 1:N
                    Y_tf_mic1{src}(j, i) = stf_j1{src}(j, i);
                end
            else
                [~, ind_min] = min(Pf(:, i, j));
                % Least dominant source: use soft masking with RAW mask
                Y_tf_mic1{ind_min}(j, i) = stf_j1{ind_min}(j, i);
                % Dominant pair: apply determined recombination
                dom_set = setdiff(1:N, ind_min);
                for idx = 1:length(dom_set)
                    src = dom_set(idx);
                    other = setdiff(dom_set, src);
                    Y_tf_mic1{src}(j, i) = X2(j, i) - c(other, j) * X1(j, i);
                    Y_tf_mic1{src}(j, i) = d(other, src, j) * Y_tf_mic1{src}(j, i);
                end
            end
        end
    end
end

for j = 1:N
    Y_tf_mic1{j}(isnan(Y_tf_mic1{j})) = 0;
    Y_tf_mic1{j}(isinf(Y_tf_mic1{j})) = 0;
end

%% ========================================================================
% MICROPHONE 2 RECONSTRUCTION (target microphone)
%% ========================================================================
c_inv = 1 ./ c;
c_inv(isinf(c_inv)) = 0;
c_inv(isnan(c_inv)) = 0;

Y_tf_mic2 = cell(N, 1);
for j = 1:N
    Y_tf_mic2{j} = zeros(F, T);
end

if N == 2
    % Determined case on mic 2
    d_inv = zeros(N, F);
    for j = 1:N
        k = setdiff(1:N, j);
        eps_val = c_inv(k, :) - c_inv(j, :);
        eps_val(1) = [];
        eps_val(end) = [];
        if ~isempty(eps_val)
            min_eps = min(abs(eps_val));
        else
            min_eps = 1e-6;
        end
        denom = c_inv(k, :) - c_inv(j, :);
        d_inv(j, :) = 1 ./ (min_eps + denom);
    end
    d_inv(isnan(d_inv)) = 0;
    d_inv(isinf(d_inv)) = 0;

    for j = 1:F
        for i = 1:T
            if Pf(1, i, j) > 1 || Pf(2, i, j) > 1
                Y_tf_mic2{1}(j, i) = stf_j2{1}(j, i);
                Y_tf_mic2{2}(j, i) = stf_j2{2}(j, i);
            else
                Y_tf_mic2{1}(j, i) = X1(j, i) - c_inv(1, j) * X2(j, i);
                Y_tf_mic2{2}(j, i) = X1(j, i) - c_inv(2, j) * X2(j, i);
                Y_tf_mic2{1}(j, i) = d_inv(1, j) * Y_tf_mic2{1}(j, i);
                Y_tf_mic2{2}(j, i) = d_inv(2, j) * Y_tf_mic2{2}(j, i);
            end
        end
    end

else
    % Underdetermined case on mic 2
    d_inv = zeros(N, N, F);
    for j = 1:N
        for k = 1:N
            if j == k, continue; end
            denom = c_inv(k, :) - c_inv(j, :);
            d_inv(j, k, :) = 1 ./ denom;
        end
    end
    d_inv(isnan(d_inv)) = 0;
    d_inv(isinf(d_inv)) = 0;

    for j = 1:F
        for i = 1:T
            if Pf(1, i, j) > 0.65 || Pf(2, i, j) > 0.65 || Pf(3, i, j) > 0.65
                for src = 1:N
                    Y_tf_mic2{src}(j, i) = stf_j2{src}(j, i);
                end
            else
                [~, ind_min] = min(Pf(:, i, j));
                Y_tf_mic2{ind_min}(j, i) = stf_j2{ind_min}(j, i);
                dom_set = setdiff(1:N, ind_min);
                for idx = 1:length(dom_set)
                    src = dom_set(idx);
                    other = setdiff(dom_set, src);
                    Y_tf_mic2{src}(j, i) = X1(j, i) - c_inv(other, j) * X2(j, i);
                    Y_tf_mic2{src}(j, i) = d_inv(other, src, j) * Y_tf_mic2{src}(j, i);
                end
            end
        end
    end
end

for j = 1:N
    Y_tf_mic2{j}(isnan(Y_tf_mic2{j})) = 0;
    Y_tf_mic2{j}(isinf(Y_tf_mic2{j})) = 0;
end
end
