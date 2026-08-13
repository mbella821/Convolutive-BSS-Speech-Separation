function Pf = cluster_em_frequency_bins(z, N, max_iter, tol)
% CLUSTER_EM_FREQUENCY_BINS Bin-wise EM clustering for TF mask estimation.
%   Pf = cluster_em_frequency_bins(z, N, max_iter, tol) runs the EM
%   algorithm independently for each frequency bin to estimate the
%   probabilistic masks (a posteriori probabilities) of N sources.
%
%   Input:
%     z        - L x T x F whitened feature tensor
%     N        - number of sources
%     max_iter - maximum EM iterations per frequency bin
%     tol      - convergence tolerance
%   Output:
%     Pf       - N x T x F tensor of probabilistic masks
%
%   The EM algorithm models the data with a complex Watson mixture model
%   (CWMM) as described in Eq. (WMM). Initialization is performed via
%   k-means on the Hermitian angle vectors (acos(abs(z))).

[L, T, F] = size(z);
Pf = zeros(N, T, F);

fprintf('\nPerforming source separation (EM clustering)...\n');

for f = 1:F
    if mod(f, 100) == 0
        fprintf('  Processing frequency bin %d / %d\n', f, F);
    end

    % Extract features at current frequency
    z_f = z(:, :, f); % L x T

    % Initialization: k-means on Hermitian angles
    teta = acos(abs(z_f)); % L x T
    [IDX, ~] = kmeans(teta', N, 'EmptyAction', 'singleton');

    % Initialize model parameters from k-means partitions
    model.A = zeros(L, N);
    for k = 1:N
        if sum(IDX == k) > 0
            a_k = mean(z_f(:, IDX == k), 2);
        else
            a_k = randn(L, 1) + 1i*randn(L, 1);
        end
        model.A(:,k) = a_k / norm(a_k);
    end

    model.sigma2 = 0.02 * ones(N,1);
    model.w = ones(N,1) / N;

    % EM iterations
    old_R = ones(N, T);
    converged = false;
    niter = 0;

    while ~converged
        niter = niter + 1;

        % Expectation step
        R = expectation(z_f, model); % N x T

        % Maximization step
        model = maximization(z_f, R);

        % Convergence check
        if norm(old_R(:) - R(:)) / norm(old_R(:)) < tol || niter >= max_iter
            converged = true;
        end
        old_R = R;
    end

    % Store posterior probabilities (probabilistic masks)
    Pf(:, :, f) = R;
end
end
