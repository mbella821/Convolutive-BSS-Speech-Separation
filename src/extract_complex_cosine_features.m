function theta = extract_complex_cosine_features(X_tf, H_ref)
% EXTRACT_COMPLEX_COSINE_FEATURES Compute complex cosine similarity features.
%   theta = extract_complex_cosine_features(X_tf, H_ref) computes the
%   complex cosine similarity between the observation vectors and the
%   provided reference vectors, as defined in Eq. (Ang) of the paper.
%
%   Input:
%     X_tf  - M x T x F observation tensor (M microphones, T frames, F bins)
%     H_ref - M x L matrix of L complex reference vectors (hard-coded)
%   Output:
%     theta - L x T x F normalized feature tensor
%


[M, T, F] = size(X_tf);
L = size(H_ref, 2);

% Pre-compute reference vector norms
abs_H = sqrt(sum(abs(H_ref).^2, 1)); % 1 x L

% Initialize feature tensor (temporary T x L x F)
theta_tmp = zeros(T, L, F);

for f = 1:F
    % Observation matrix at frequency f: M x T
    X_f = X_tf(:, :, f);

    % Normalize observation vectors to unit norm
    norm_X = sqrt(sum(abs(X_f).^2, 1)); % 1 x T
    norm_X(norm_X == 0) = 1; % avoid division by zero
    X_norm = bsxfun(@rdivide, X_f, norm_X); % M x T

    % Complex cosine similarity: (M x T)' * (M x L) -> T x L
    cos_sim = (X_norm' * H_ref); % T x L
    cos_sim = bsxfun(@rdivide, cos_sim, abs_H); % T x L

    % Normalize each feature vector to unit norm per time frame
    norm_cos = sqrt(sum(abs(cos_sim).^2, 2)); % T x 1
    norm_cos(norm_cos == 0) = 1;
    cos_sim = bsxfun(@rdivide, cos_sim, norm_cos); % T x L

    theta_tmp(:, :, f) = cos_sim;
end

% Permute to L x T x F for subsequent processing
theta = permute(theta_tmp, [2, 1, 3]);

% Safeguard against NaN values
theta(isnan(theta)) = 0;
end
