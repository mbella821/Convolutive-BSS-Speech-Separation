# Blind Source Separation via RTF-based Recombination

MATLAB implementation of the method proposed in:
> **"Underdetermined Convolutive Blind Source Separation of Speech Signals Using Relaxed Time–Frequency Sparsity and RTF‑Based Recombination"**

This repository provides a clean, modular, and unified implementation that handles **both determined and underdetermined** convolutive mixtures of speech signals.

## Overview

The method consists of three main steps:
1. **TF Mask Estimation**: Complex cosine similarity between observation vectors and reference vectors is used as a clustering feature. An EM-type algorithm estimates probabilistic masks per frequency bin.
2. **RTF Estimation**: Relative Transfer Functions (RTFs) are blindly estimated from the most reliable single-source TF bins.
3. **Source Reconstruction**: A novel RTF-based recombination strategy reconstructs source spatial images, reducing artifacts caused by classical TF masking.

## Repository Structure

```
BSS_RTF_Recombination/
├── main.m                              % Main script
├── src/
│   ├── compute_stft.m                  % STFT analysis
│   ├── compute_istft.m                 % ISTFT synthesis
│   ├── extract_complex_cosine_features.m % Feature extraction 
│   ├── whiten_features.m               % Whitening & normalization
│   ├── cluster_em_frequency_bins.m     % Bin-wise EM clustering
│   ├── align_permutation.m             % Global + local permutation alignment
│   ├── estimate_rtf.m                  % Blind RTF estimation 
│   ├── reconstruct_sources.m           % RTF-based recombination 
│   └── evaluate_separation.m           % BSS evaluation wrapper
└── utils/
    ├── expectation.m                   % EM Expectation step
    ├── maximization.m                  % EM Maximization step
    ├── loggausspdf.m                   % Complex Gaussian log-pdf
    └── logsumexp.m                     % Numerically stable log-sum-exp
```

## Requirements

- MATLAB (R2016b or later recommended)
- Signal Processing Toolbox (for `kmeans`, `fft`, `ifft`)


## Usage

1. Place your audio files and mixing filters in the working directory.
2. Open `main.m` and adjust the parameters (number of sources `N`, microphones `M`, file paths, etc.).
3. Run `main.m`.

## Parameters

Key parameters are grouped in the `params` structure inside `main.m`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `window_size` | STFT window length | 2048 |
| `nfft` | FFT size | 2048 |
| `overlap` | Overlap in samples | 512 |
| `num_sources` | Number of sources N | 2 or 3 |
| `num_mics` | Number of microphones M | 2 |
| `max_iter_em` | Max EM iterations per bin | 100 |
| `tol_em` | EM convergence tolerance | 1e-7 |
| `eta` | Threshold for dominant TF bin selection (RTF) | 0.95 |
| `mask_threshold` | Threshold for dominant source detection | 0.65 |

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{bella2026underdetermined,
  title={Underdetermined convolutive blind source separation of speech signals using relaxed time--frequency sparsity and RTF-based recombination},
  author={Bella, Mostafa and Saylani, Hicham and Hosseini, Shahram},
  journal={Computer Speech \& Language},
  pages={101992},
  year={2026},
  publisher={Elsevier}
}
```

## License

This code is provided for academic and research purposes.
