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
addpath(genpath('/home/mostafa/Mostafa/sawada/CTFMR version finale/src'))
addpath(genpath('/home/mostafa/Mostafa/sawada/CTFMR version finale/utils'))
addpath(genpath('/home/mostafa/Mostafa/sawada'))

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

% 
% % Reference vectors for complex cosine similarity (optional)
% % If empty, M orthogonal complex vectors are generated automatically.
% Ref(:,1)=[3 ;
%    1 ];
% 
% Ref(:,2)=[1;
%   -3];
% params.reference_vectors = Ref;

%% ========================================================================
% 2. LOAD AUDIO AND MIXING FILTERS
%% ========================================================================
% NOTE: Adapt the file paths below to your local dataset.
%
% Example for N=2 sources:
%   [s1, FS] = audioread('male_s4.wav');
%   [s2, FS] = audioread('female_s5.wav');
%   a = load('filters/2src_50ms_1m_1m_config2.mat');
%
% Example for N=3 sources:
%   [s1, FS] = audioread('s2.wav');
%   [s2, FS] = audioread('s4.wav');
%   [s3, FS] = audioread('male_s2.wav');
%   a = load('filters/3src_100ms_1m_1m_config2.mat');

fprintf('Loading audio signals and mixing filters...\n');

% -------------------------------------------------------------------------
% USER CONFIGURATION: replace with your actual file paths
% -------------------------------------------------------------------------
if params.num_sources == 2
    [s1, FS] = audioread('/home/mostafa/Mostafa/sawada/male_s4.wav');
    [s2, FS] = audioread('/home/mostafa/Mostafa/sawada/female_s5.wav');
    s1 = s1(1:params.signal_length);
    s2 = s2(1:params.signal_length);
    sources = [s1, s2];

    a = load('/home/mostafa/Mostafa/sawada/filters/2src_50ms_1m_1m_config2.mat');
elseif params.num_sources == 3
    [s1, FS] = audioread('s2.wav');
    [s2, FS] = audioread('s4.wav');
    [s3, FS] = audioread('male_s2.wav');
    s1 = s1(1:params.signal_length);
    s2 = s2(1:params.signal_length);
    s3 = s3(1:params.signal_length);
    sources = [s1, s2, s3];

    a = load('/home/mostafa/Mostafa/sawada/filters/3src_100ms_1m_1m_config2.mat');
else
    error('This demo supports N=2 or N=3 sources. Please adapt main.m for other cases.');
end

% Extract mixing filters
A = a.A; % M x N x filter_length
b = cell(params.num_mics, params.num_sources);
for i = 1:params.num_mics
    for j = 1:params.num_sources
        b{i,j} = squeeze(A(i,j,:));
    end
end

%% ========================================================================
% 3. GENERATE CONVOLUTIVE MIXTURES
%% ========================================================================
fprintf('Generating convolutive mixtures...\n');
x = zeros(params.signal_length, params.num_mics);
for i = 1:params.num_mics
    for j = 1:params.num_sources
        x(:, i) = x(:, i) + fftfilt(b{i,j}, sources(:, j));
    end
end

%% ========================================================================
% 5. STFT ANALYSIS (positive frequencies only)
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
% 6. FEATURE EXTRACTION (Complex Cosine Similarity)
%% ========================================================================
fprintf('Extracting complex cosine similarity features...\n');
theta = extract_complex_cosine_features(X_tf, H_ref);

%% ========================================================================
% 7. WHITENING AND NORMALIZATION
%% ========================================================================
fprintf('Whitening features...\n');
z = whiten_features(theta);

%% ========================================================================
% 8. EM CLUSTERING (Bin-wise)
%% ========================================================================
Pf = cluster_em_frequency_bins(z, params.num_sources, params.max_iter_em, params.tol_em);

%% ========================================================================
% 9. PERMUTATION ALIGNMENT
%% ========================================================================
Pf = align_permutation(Pf, params.max_iter_perm, params.tol_perm);

%% ========================================================================
% 10. SOURCE RECONSTRUCTION (both microphones, exact original logic)
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
% 11. EVALUATION (BSSeval)
%% ========================================================================
fprintf('\nEvaluating separation performance...\n');

% True spatial images
s_img = zeros(params.num_sources, params.signal_length, params.num_mics);
for j = 1:params.num_sources
    for i = 1:params.num_mics
        s_img(j, :, i) = fftfilt(b{i,j}, sources(:, j));
        s_img(j, :, i) = s_img(j, :, i) / norm(s_img(j, :, i));
    end
end

% Estimated spatial images (exact original ordering)
se_img = zeros(params.num_sources, params.signal_length, params.num_mics);
for j = 1:params.num_sources
    se_img(j, :, 1) = y_mic1_norm(j, :); % mic 1
    se_img(j, :, 2) = y_mic2_norm(j, :); % mic 2
end

if exist('bss_eval_images', 'file')
    [SDR, ISR, SIR, SAR, perm] = evaluate_separation(se_img, s_img);
else
    fprintf('BSSeval toolbox not found. Skipping evaluation.\n');
    fprintf('Install from: https://bass-db.gforge.inria.fr/bss_eval/\n');
end

%% ========================================================================
% 12. SAVE RESULTS
%% ========================================================================
% fprintf('\nSaving separated signals...\n');
% for j = 1:params.num_sources
%     audiowrite(sprintf('separated_source_%d_mic1.wav', j), y_mic1(j, :)', FS);
%     audiowrite(sprintf('separated_source_%d_mic2.wav', j), y_mic2(j, :)', FS);
% end

fprintf('\nDone.\n');
