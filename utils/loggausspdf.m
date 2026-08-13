function logP = loggausspdf(X, Sigma)
% LOGGAUSSPDF Logarithm of 0-mean complex-circular Gaussian pdf.
%   logP = loggausspdf(X, Sigma) computes the log-probability density
%   values for each of the T M-dimensional complex datapoints in X under
%   a zero-mean complex-circular Gaussian with covariance Sigma.
%
%   Input:
%     X     - M x T matrix of T complex input vectors
%     Sigma - M x M covariance matrix
%   Output:
%     logP  - 1 x T vector of log probability density values
%
%   See Eq. (WMM) in the paper for the underlying statistical model.

M = size(X,1);

[U,p] = chol(Sigma); % U is upper triangular (M x M)

if p ~= 0
    error('ERROR: Sigma is not positive definite.');
end

Q = U' \ X; % Weighted data (M x T)
q = -dot(Q,Q,1);  % (1 x T) Quadratic term

% Normalization constant for complex Gaussian with isotropic covariance
c = -(M-1)*(log(pi)+log(Sigma(1,1)));

logP = c + q; % (1 x T)
end
