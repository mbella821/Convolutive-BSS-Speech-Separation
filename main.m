%% MAIN Main script for Blind Source Separation via RTF-based Recombination.
%   This script demonstrates the complete pipeline of the proposed method
%   for both determined and underdetermined convolutive mixtures.
%
%   The method consists of four steps:
%     1. STFT analysis of the microphone signals.
%     2. Complex cosine similarity feature extraction and EM clustering
%        to estimate probabilistic TF masks.
%     3. Permutation alignment across frequency bins.
%     4. Blind RTF estimation and source recombination.
%
%   Author: Mostafa Bella.
%   Date: 2024

clear; clc; close all;
addpath(genpath('./src'))
addpath(genpath('./utils'))

%% ========================================================================
% 1. PARAMETERS (exactly as original)
%% ========================================================================
params.window_size   = 1024*2;      % STFT window length
params.nfft          = 1024*2;      % FFT size
params.overlap       = 0.25 * params.window_size; % overlap in samples
params.num_sources   = 2;           % Number of sources N (2 or 3)
params.num_mics      = 2;           % Number of microphones M (must be 2)
params.signal_length = 160000;      % Signal length in samples

% Number of positive-frequency bins (DC to Nyquist)
params.num_freq_bins = floor(params.nfft/2) + 1;

% EM clustering parameters
params.max_iter_em   = 100;
params.tol_em        = 1e-7;

% Permutation alignment parameters
params.max_iter_perm = 50;
params.tol_perm      = 1e-10;

% Reconstruction threshold (determined: >1 always false, underdetermined: 0.65)
if params.num_sources == 2
    params.mask_threshold = 1.0;
else
    params.mask_threshold = 0.65;
end

%% ========================================================================
% 2. HARD-CODED REFERENCE VECTORS (exactly as original)
%% ========================================================================
% These are the exact reference vectors used in the original code.
% For N=2: Ref3 and Ref6 are used.
% For N=3: Ref3 and Ref4 are used.

Ref1 = 1*ones(2,1)*1 + 1*1i;
Ref2 = [0.3245 + 0.4221*1i; 0.2498 + 0.2067*1i];
Ref3 = [1; -3];
Ref4 = [3 + 3*1i; 1 + 1*1i];
Ref5 = [1 - 2*1i; 0.5 + 2*1i];
Ref6 = [3; 1];

if params.num_sources == 2
    H_ref = [Ref3, Ref6];  % exact original for 2 sources
elseif params.num_sources == 3
    H_ref = [Ref3, Ref4];  % exact original for 3 sources
else
    error('Only N=2 or N=3 are supported in this demo.');
end

%% ========================================================================
% 2. LOAD AUDIO AND MIXING FILTERS
%% ========================================================================
% NOTE: Adapt the file paths below to your local dataset.

fprintf('Loading audio signals...\n');

% -------------------------------------------------------------------------
% USER CONFIGURATION: replace with your actual file paths
% -------------------------------------------------------------------------
if params.num_sources == 2
    % Determined case: load mixed microphone signals
    [mix1, FS] = audioread('./data/sample_audio/deter_mix_mic1.wav');
    [mix2, FS] = audioread('./data/sample_audio/deter_mix_mic2.wav');
    
    mix1 = mix1(1:min(params.signal_length, length(mix1)));
    mix2 = mix2(1:min(params.signal_length, length(mix2)));
    
    % Pad if necessary
    if length(mix1) < params.signal_length
        mix1 = [mix1; zeros(params.signal_length - length(mix1), 1)];
    end
    if length(mix2) < params.signal_length
        mix2 = [mix2; zeros(params.signal_length - length(mix2), 1)];
    end
    
    x = [mix1, mix2];  % M x signal_length
    
elseif params.num_sources == 3
    % Underdetermined case: load mixed microphone signals
    [mix1, FS] = audioread('./data/sample_audio/under_mix_mic1.wav');
    [mix2, FS] = audioread('./data/sample_audio/under_mix_mic2.wav');
    
    mix1 = mix1(1:min(params.signal_length, length(mix1)));
    mix2 = mix2(1:min(params.signal_length, length(mix2)));
    
    % Pad if necessary
    if length(mix1) < params.signal_length
        mix1 = [mix1; zeros(params.signal_length - length(mix1), 1)];
    end
    if length(mix2) < params.signal_length
        mix2 = [mix2; zeros(params.signal_length - length(mix2), 1)];
    end
    
    x = [mix1, mix2];  % M x signal_length
else
    error('This demo supports N=2 or N=3 sources. Please adapt main.m for other cases.');
end

fprintf('Audio loaded: %d samples, %d kHz sampling rate\n', params.signal_length, FS/1000);

%% ========================================================================
% 3. STFT ANALYSIS (positive frequencies only)
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
    X_tf(i, :, :) = S'; % M x T x F_pos
end

[~, T, F] = size(X_tf);
fprintf('  STFT size: %d mics x %d frames x %d bins\n', params.num_mics, T, F);

%% ========================================================================
% 4. FEATURE EXTRACTION (Complex Cosine Similarity)
%% ========================================================================
fprintf('Extracting complex cosine similarity features...\n');
theta = extract_complex_cosine_features(X_tf, H_ref);

%% ========================================================================
% 5. WHITENING AND NORMALIZATION
%% ========================================================================
fprintf('Whitening features...\n');
z = whiten_features(theta);

%% ========================================================================
% 6. EM CLUSTERING (Bin-wise)
%% ========================================================================
Pf = cluster_em_frequency_bins(z, params.num_sources, params.max_iter_em, params.tol_em);

%% ========================================================================
% 7. PERMUTATION ALIGNMENT
%% ========================================================================
Pf = align_permutation(Pf, params.max_iter_perm, params.tol_perm);

%% ========================================================================
% 8. SOURCE RECONSTRUCTION (both microphones, exact original logic)
%% ========================================================================
fprintf('\nReconstructing source spatial images...\n');
[Y_tf_mic1, Y_tf_mic2] = reconstruct_sources(X_tf, Pf, params);

% ISTFT to time domain
y_mic1 = zeros(params.num_sources, params.signal_length);
y_mic2 = zeros(params.num_sources, params.signal_length);
for j = 1:params.num_sources
    y_mic1(j, :) = compute_istft(Y_tf_mic1{j}, window, params.overlap, params.signal_length, params.nfft);
    y_mic2(j, :) = compute_istft(Y_tf_mic2{j}, window, params.overlap, params.signal_length, params.nfft);
end

% Normalize (as original)
y_mic1_norm = zeros(params.num_sources, params.signal_length);
y_mic2_norm = zeros(params.num_sources, params.signal_length);
for j = 1:params.num_sources
    y_mic1_norm(j, :) = y_mic1(j, :) / norm(y_mic1(j, :));
    y_mic2_norm(j, :) = y_mic2(j, :) / norm(y_mic2(j, :));
end

%% ========================================================================
% 9. SAVE RESULTS
%% ========================================================================
fprintf('\nSaving separated signals...\n');
output_dir = './results';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

for j = 1:params.num_sources
    audiowrite(sprintf('%s/separated_source_%d_mic1.wav', output_dir, j), y_mic1(j, :)', FS);
    audiowrite(sprintf('%s/separated_source_%d_mic2.wav', output_dir, j), y_mic2(j, :)', FS);
    fprintf('  Saved: separated_source_%d_mic1.wav, separated_source_%d_mic2.wav\n', j, j);
end

fprintf('\nDone! Separated signals saved to ''%s/'' directory.\n', output_dir);
