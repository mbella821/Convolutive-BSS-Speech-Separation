function Pf = align_permutation(Pf, max_iter, tol)
% ALIGN_PERMUTATION Solve the permutation problem across frequencies.
%   Pf = align_permutation(Pf, max_iter, tol) aligns the source indices
%   across frequency bins using a two-stage approach: global optimization
%   followed by local optimization, as proposed in Sawada et al. 2010.
%
%   Input:
%     Pf       - N x T x F unaligned probabilistic mask tensor
%     max_iter - maximum iterations for each stage
%     tol      - convergence tolerance
%   Output:
%     Pf       - N x T x F permutation-aligned mask tensor

[N, T, F] = size(Pf);
P = perms(1:N);
nperms = size(P,1);

%% Stage 1: Global optimization with a single centroid
fprintf('\nGlobal permutation alignment...\n');
niter = 0;
converged = false;
old_c = ones(N, T);

while ~converged
    niter = niter + 1;
    c = mean(Pf, 3)'; % T x N (centroid across frequencies)

    Pf_permut = zeros(N, T, F);
    for f = 1:F
        score_q = zeros(nperms, 1);
        for i = 1:nperms
            q = corr(Pf(P(i,:), :, f)', c);
            score_q(i) = 2*sum(diag(q)) - sum(sum(q));
        end
        [~, ibest] = max(score_q);
        Pf_permut(:, :, f) = Pf(P(ibest, :), :, f);
    end

    % Convergence criterion based on first source trajectory
    if norm(squeeze(Pf(1,:,:)) - squeeze(Pf_permut(1,:,:))) / norm(squeeze(Pf(1,:,:))) < tol || niter >= max_iter
        converged = true;
    end

    old_c = c;
    Pf = Pf_permut;
end

%% Stage 2: Local optimization using neighboring frequencies
fprintf('\nLocal permutation alignment...\n');
niter = 0;
converged = false;
Pf2 = Pf;
Pf_permut2 = zeros(N, T, F);

while ~converged
    niter = niter + 1;

    for f = 4:F-3
        score_qq = zeros(nperms, 12);
        for i = 1:nperms
            % Neighboring frequencies (left)
            q11 = corr(Pf(P(i,:), :, f)', Pf(:, :, f-1)');
            score_qq(i,1) = 2*sum(diag(q11)) - sum(sum(q11));
            q12 = corr(Pf(P(i,:), :, f)', Pf(:, :, f-2)');
            score_qq(i,2) = 2*sum(diag(q12)) - sum(sum(q12));
            q13 = corr(Pf(P(i,:), :, f)', Pf(:, :, f-3)');
            score_qq(i,3) = 2*sum(diag(q13)) - sum(sum(q13));

            % Neighboring frequencies (right)
            q21 = corr(Pf(P(i,:), :, f)', Pf(:, :, f+1)');
            score_qq(i,4) = 2*sum(diag(q21)) - sum(sum(q21));
            q22 = corr(Pf(P(i,:), :, f)', Pf(:, :, f+2)');
            score_qq(i,5) = 2*sum(diag(q22)) - sum(sum(q22));
            q23 = corr(Pf(P(i,:), :, f)', Pf(:, :, f+3)');
            score_qq(i,6) = 2*sum(diag(q23)) - sum(sum(q23));

            % Harmonic relation (half frequency)
            q31 = corr(Pf(P(i,:), :, f)', Pf(:, :, round(f/2))');
            score_qq(i,7) = 2*sum(diag(q31)) - sum(sum(q31));
            q32 = corr(Pf(P(i,:), :, f)', Pf(:, :, round(f/2)-1)');
            score_qq(i,8) = 2*sum(diag(q32)) - sum(sum(q32));
            q33 = corr(Pf(P(i,:), :, f)', Pf(:, :, round(f/2)+1)');
            score_qq(i,9) = 2*sum(diag(q33)) - sum(sum(q33));

            % Harmonic relation (double frequency)
            if f < F/2
                q41 = corr(Pf(P(i,:), :, f)', Pf(:, :, 2*f)');
                score_qq(i,10) = 2*sum(diag(q41)) - sum(sum(q41));
                q42 = corr(Pf(P(i,:), :, f)', Pf(:, :, 2*f-1)');
                score_qq(i,11) = 2*sum(diag(q42)) - sum(sum(q42));
                q43 = corr(Pf(P(i,:), :, f)', Pf(:, :, 2*f+1)');
                score_qq(i,12) = 2*sum(diag(q43)) - sum(sum(q43));
            end
        end

        score_qq(isnan(score_qq)) = 0;
        sum_score_qq = sum(score_qq, 2);
        [~, ibest] = max(sum_score_qq);
        Pf_permut2(:, :, f) = Pf(P(ibest, :), :, f);
    end

    % Keep boundary frequencies unchanged 
    for f = 1:3
        Pf_permut2(:, :, f) = Pf2(:, :, f);
    end
    for f = F-3:F
        Pf_permut2(:, :, f) = Pf2(:, :, f);
    end

    % Convergence check
    if norm(squeeze(Pf(1,:,:)) - squeeze(Pf_permut2(1,:,:))) / norm(squeeze(Pf(1,:,:))) < tol || niter >= 15
        converged = true;
    end

    Pf = Pf_permut2;
end
end
