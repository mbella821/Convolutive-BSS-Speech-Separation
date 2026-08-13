function model = maximization(X, R)
% MAXIMIZATION Maximization step of the EM algorithm for CWMM.
%   model = maximization(X, R) updates the cluster parameters by
%   maximizing the expected complete-data log-likelihood.
%
%   Input:
%     X     - M x T matrix of T complex feature vectors
%     R     - N x T posterior probabilities from the E-step
%   Output:
%     model - updated struct with fields .A, .sigma2, .w
%
%   See Appendix B of the paper for the derivation of the update rules.

N = size(R,1);
T = size(R,2);
phi = 1000; % Dirichlet regularization hyper-parameter
M = size(X,1);

model.sigma2 = zeros(N,1);
model.A = zeros(M,N);

% Update cluster weights (regularized)
model.w = (sum(R,2) + phi - 1) / (T + N*(phi - 1)); % N x 1

for k = 1:N
    sumRk = sum(R(k,:));

    % Weighted data
    r_k = sqrt(R(k,:) / sumRk); % 1 x T
    X_k = bsxfun(@times, X, r_k); % M x T

    % Weighted sample covariance matrix
    Gamma_k = X_k * X_k'; % M x M

    % Leading eigenvector of Gamma_k (steering vector update)
    [v_k, ~, ~] = svd(Gamma_k);
    v_k = v_k(:,1);
    model.A(:,k) = v_k / norm(v_k);

    % Subtract projection of X on updated centroid
    mu = X - bsxfun(@times, model.A(:,k)'*X, model.A(:,k)); % M x T

    % Update noise variance
    model.sigma2(k) = sum(sum(abs(mu).^2,1) .* R(k,:)) / ((M-1) * sumRk);
end

% Regularize noise variance to avoid numerical problems
model.sigma2(model.sigma2 < 1e-6) = 1e-6;
end
