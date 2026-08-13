%% MAIN Main script for "Underdetermined Convolutive Blind Source Separation
%  of Speech Signals Using Relaxed Time-Frequency Sparsity and
%  RTF-Based Recombination.
%
%  This script demonstrates the complete processing pipeline of the
%  proposed method for both determined and underdetermined convolutive
%  mixtures.
%
%  The processing pipeline consists of the following steps:
%    1. STFT analysis of the microphone signals.
%    2. Complex cosine similarity feature extraction followed by
%       EM-based clustering to estimate probabilistic time-frequency
%       masks.
%    3. Permutation alignment across frequency bins.
%    4. Blind RTF estimation and source recombination.
%
%  Author: Mostafa Bella.

clear; clc; close all;

%% Add project directories to MATLAB path
project_root = fileparts(mfilename('fullpath'));

addpath(genpath(fullfile(project_root, 'src')));
addpath(genpath(fullfile(project_root, 'utils')));

data_dir = fullfile(project_root, 'data');

%% ========================================================================
% 1. PARAMETERS
%% ========================================================================

params.window_size   = 1024*2;      % STFT window length
params.nfft          = 1024*2;      % FFT size
params.overlap       = 0.25 * params.window_size; % Overlap in samples
params.num_sources   = 3;            % Number of sources (2 or 3)
params.num_mics      = 2;            % Number of microphones (must be 2)
params.signal_length = 160000;       % Signal length in samples

% Number of positive-frequency bins (DC to Nyquist)
params.num_freq_bins = floor(params.nfft/2) + 1;

% EM clustering parameters
params.max_iter_em   = 100;          % Maximum number of EM iterations
params.tol_em        = 1e-7;         % EM convergence tolerance

% Permutation alignment parameters
params.max_iter_perm = 50;           % Maximum number of permutation iterations
params.tol_perm      = 1e-10;        % Permutation convergence tolerance

% Reconstruction threshold:
%   - Determined case (2 sources): 1.0
%   - Underdetermined case (3 sources): 0.65
if params.num_sources == 2
    params.mask_threshold = 1.0;
else
    params.mask_threshold = 0.65;
end

%% ========================================================================
% 2. REFERENCE VECTORS
%% ========================================================================

fprintf('Reference vectors...\n');

% Reference vectors used for complex cosine similarity.
% If no reference vectors are provided, M orthogonal complex vectors
% are generated automatically.
params.reference_vectors = [];

if isempty(params.reference_vectors)

    % Generate M orthogonal complex reference vectors (L = M).
    L = params.num_mics;
    H_ref = (randn(params.num_mics, L) + 1i*randn(params.num_mics, L)) / sqrt(2);
    [H_ref, ~] = qr(H_ref);
    H_ref = H_ref(:, 1:L);

else

    H_ref = params.reference_vectors;

end

%% ========================================================================
% 3. LOAD AUDIO MIXTURES
%% ========================================================================

% Load the appropriate mixture depending on the number of sources.
% The input files should be adapted to the local dataset if necessary.
fprintf('Loading audio mixtures...\n');

% -------------------------------------------------------------------------
% USER CONFIGURATION:
% Replace the file paths below with the paths to the desired mixtures.
% -------------------------------------------------------------------------

if params.num_sources == 2

    % Determined mixture: 2 sources and 2 microphones.
    [x1, FS] = audioread('deter_mix_mic1.wav');
    [x2, ~]  = audioread('deter_mix_mic2.wav');
    x = [x1, x2];

elseif params.num_sources == 3

    % Underdetermined mixture: 3 sources and 2 microphones.
    [x1, ~] = audioread('under_mix_mic1.wav');
    [x2, ~] = audioread('under_mix_mic2.wav');
    x = [x1, x2];

else

    error('This demo supports N=2 or N=3 sources. Please adapt main.m for other cases.');

end

%% ========================================================================
% 4. STFT ANALYSIS
%% ========================================================================

fprintf('Computing one-sided STFT...\n');

window = hanning(params.window_size);
params.window = window;

F_pos = params.num_freq_bins;
X_tf = zeros(params.num_mics, [], F_pos);

for i = 1:params.num_mics

    S = compute_stft(x(:, i), window, params.overlap, params.nfft);

    if i == 1
        T = size(S, 2);
        X_tf = zeros(params.num_mics, T, F_pos);
    end

    X_tf(i, :, :) = S'; % Microphones x time frames x frequency bins

end

[~, T, F] = size(X_tf);

fprintf('  STFT size: %d mics x %d frames x %d bins\n', ...
        params.num_mics, T, F);

%% ========================================================================
% 5. FEATURE EXTRACTION: COMPLEX COSINE SIMILARITY
%% ========================================================================

fprintf('Extracting complex cosine similarity features...\n');

theta = extract_complex_cosine_features(X_tf, H_ref);

%% ========================================================================
% 6. WHITENING AND NORMALIZATION
%% ========================================================================

fprintf('Whitening features...\n');

z = whiten_features(theta);

%% ========================================================================
% 7. EM CLUSTERING
%% ========================================================================

% Perform EM clustering independently across frequency bins to estimate
% probabilistic source masks.
Pf = cluster_em_frequency_bins( ...
    z, ...
    params.num_sources, ...
    params.max_iter_em, ...
    params.tol_em);

%% ========================================================================
% 8. PERMUTATION ALIGNMENT
%% ========================================================================

% Resolve the frequency-dependent permutation ambiguity between clusters.
Pf = align_permutation( ...
    Pf, ...
    params.max_iter_perm, ...
    params.tol_perm);

%% ========================================================================
% 9. SOURCE RECONSTRUCTION
%% ========================================================================

fprintf('\nReconstructing source spatial images...\n');

% Reconstruct the spatial images of the estimated sources at both
% microphone channels.
[Y_tf_mic1, Y_tf_mic2] = reconstruct_sources(X_tf, Pf, params);

% Convert the estimated source images back to the time domain using ISTFT.
y_mic1 = zeros(params.num_sources, params.signal_length);
y_mic2 = zeros(params.num_sources, params.signal_length);

for j = 1:params.num_sources

    y_mic1(j, :) = compute_istft( ...
        Y_tf_mic1{j}, ...
        window, ...
        params.overlap, ...
        params.signal_length, ...
        params.nfft);

    y_mic2(j, :) = compute_istft( ...
        Y_tf_mic2{j}, ...
        window, ...
        params.overlap, ...
        params.signal_length, ...
        params.nfft);

end

% Normalize the reconstructed source images independently for each
% microphone channel.
y_mic1_norm = zeros(params.num_sources, params.signal_length);
y_mic2_norm = zeros(params.num_sources, params.signal_length);

for j = 1:params.num_sources

    y_mic1_norm(j, :) = y_mic1(j, :) / norm(y_mic1(j, :));
    y_mic2_norm(j, :) = y_mic2(j, :) / norm(y_mic2(j, :));

end

%% ========================================================================
% 10. SAVE RESULTS
%% ========================================================================

fprintf('\nSaving separated signals...\n');

for j = 1:params.num_sources

    audiowrite( ...
        sprintf('separated_source_%d_mic1.wav', j), ...
        y_mic1(j, :)', ...
        FS);

    audiowrite( ...
        sprintf('separated_source_%d_mic2.wav', j), ...
        y_mic2(j, :)', ...
        FS);

end

fprintf('\nDone.\n');