function R = expectation(X, model)
% EXPECTATION Expectation step of the EM algorithm for CWMM.
%   R = expectation(X, model) computes the posterior probabilities
%   p(z_n = k | x_n; model) for each cluster k and each time frame.
%
%   Input:
%     X     - M x T matrix of T complex feature vectors
%     model - struct with fields:
%               .A      - M x N steering vectors (centroids)
%               .sigma2 - N x 1 noise variances
%               .w      - N x 1 cluster weights
%   Output:
%     R     - N x T posterior probability matrix
%
%   This corresponds to the E-step described in Section "Time-Frequency
%   mask estimation" of the paper.

N = numel(model.w);
[M,T] = size(X);

logR = zeros(N,T);

% Compute non-normalized posterior probabilities
for k = 1:N
    a_k = model.A(:,k);

    % Subtract projection of X on a_k (orthogonal deviation)
    mu = X - bsxfun(@times, a_k'*X, a_k); % M x T

    % Covariance matrix (isotropic noise)
    Sigma_k = model.sigma2(k) * eye(M);

    % Log posterior
    logR(k,:) = log(model.w(k)) + loggausspdf(mu, Sigma_k); % 1 x T
end

% Normalize the posterior probabilities (softmax in log domain)
logR = bsxfun(@minus, logR, logsumexp(logR,1)); % N x T
R = exp(logR); % N x T
end
