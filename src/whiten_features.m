function z = whiten_features(theta)
% WHITEN_FEATURES Whitening and re-normalization of feature vectors.
%   z = whiten_features(theta) applies whitening and unit-norm
%   re-normalization to the complex cosine similarity features, producing
%   the processed feature vectors Z(t) used for EM clustering.
%
%   Input:
%     theta - L x T x F feature tensor
%   Output:
%     z     - L x T x F whitened and re-normalized feature tensor
%
%   The whitening matrix is W = D^{-1/2} * E^H, where E and D are the
%   eigenvectors and eigenvalues of the covariance matrix of the normalized
%   features. See Section "Time-Frequency mask estimation" of the paper.

[L, T, F] = size(theta);
z = zeros(L, T, F);

for f = 1:F
    % Covariance matrix of normalized features at frequency f
    C = theta(:,:,f) * theta(:,:,f)'; % L x L

    % Eigen-decomposition
    [E, D] = eig(C);

    % Whitening matrix
    V = D^(-0.5) * E'; % L x L

    % Apply whitening
    z_f = V * theta(:,:,f); % L x T

    % Re-normalize to unit norm per time frame
    normZ = sqrt(sum(abs(z_f).^2, 1)); % 1 x T
    normZ(normZ == 0) = 1;
    z_f = bsxfun(@rdivide, z_f, normZ); % L x T

    z(:,:,f) = z_f;
end

% Safeguard against numerical issues
z(isnan(z)) = 0;
end
