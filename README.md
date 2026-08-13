# Convolutive Blind Source Separation - Speech Signals

MATLAB implementation of the method proposed in:

> **"Underdetermined Convolutive Blind Source Separation of Speech Signals Using Relaxed Time–Frequency Sparsity and RTF‑Based Recombination"**
>
> **Bella, M.**, Saylani, H., & Hosseini, S. (2026)
>
> *Computer Speech & Language*, vol. 101, p. 101992
>
> [DOI: 10.1016/j.csl.2026.101992](http://doi.org/10.1016/j.csl.2026.101992)

This repository provides a clean, modular, and unified MATLAB implementation that handles **both determined and underdetermined** convolutive mixtures of speech signals.

---

## 🎯 Overview

The method consists of three main steps:

1. **TF Mask Estimation**: Complex cosine similarity between observation vectors and reference vectors is used as a clustering feature. An EM-type algorithm estimates probabilistic masks per frequency bin.

2. **RTF Estimation**: Relative Transfer Functions (RTFs) are blindly estimated from the most reliable single-source TF bins.

3. **Source Reconstruction**: A novel RTF-based recombination strategy reconstructs source spatial images, reducing artifacts caused by classical TF masking.

---

## 📁 Repository Structure

```
Convolutive-BSS-Speech-Separation/
├── README.md                           % This file
├── main.m                              % Main script
├── .gitignore                          % Git ignore file
├── src/                                % Core algorithm functions
│   ├── compute_stft.m                  % STFT analysis
│   ├── compute_istft.m                 % ISTFT synthesis
│   ├── extract_complex_cosine_features.m % Feature extraction
│   ├── whiten_features.m               % Whitening & normalization
│   ├── cluster_em_frequency_bins.m     % Bin-wise EM clustering
│   ├── align_permutation.m             % Permutation alignment
│   ├── estimate_rtf.m                  % Blind RTF estimation
│   ├── reconstruct_sources.m           % RTF-based recombination
│   └── evaluate_separation.m           % BSS evaluation
├── utils/                              % Utility functions
│   ├── expectation.m                   % EM Expectation step
│   ├── maximization.m                  % EM Maximization step
│   ├── loggausspdf.m                   % Complex Gaussian log-pdf
│   └── logsumexp.m                     % Numerically stable log-sum-exp
└── data/                               % Sample audio data & filters
    ├── README.md                       % Data documentation
    ├── sample_audio/
    │   ├── deter_mix_mic1.wav          % Determined case - microphone 1
    │   ├── deter_mix_mic2.wav          % Determined case - microphone 2
    │   ├── under_mix_mic1.wav          % Underdetermined case - microphone 1
    │   └── under_mix_mic2.wav          % Underdetermined case - microphone 2
```

---

## ⚙️ Requirements

- **MATLAB** (R2016b or later recommended)
- **Signal Processing Toolbox** (for FFT, window functions, etc.)
- **BSS Eval Toolbox** (optional, for performance evaluation)
  - Download from: [https://bass-db.gforge.inria.fr/bss_eval/](https://bass-db.gforge.inria.fr/bss_eval/)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/mbella821/Convolutive-BSS-Speech-Separation.git
cd Convolutive-BSS-Speech-Separation
```

### 2. Add Paths to MATLAB

```matlab
addpath(genpath('./src'));
addpath(genpath('./utils'));
```

### 3. Run the Main Script

```matlab
main
```

The script will:
- Load sample audio mixtures from `data/sample_audio/`
- Apply STFT analysis
- Extract complex cosine similarity features
- Perform EM clustering for TF mask estimation
- Align permutations across frequency bins
- Estimate RTFs
- Reconstruct separated sources
- Evaluate separation performance (if BSSeval is available)

---

## 📝 Parameters

Key parameters are configured in `main.m`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `window_size` | STFT window length | 2048 |
| `nfft` | FFT size | 2048 |
| `overlap` | Overlap in samples | 512 |
| `num_sources` | Number of sources N | 2 or 3 |
| `num_mics` | Number of microphones M | 2 |
| `max_iter_em` | Max EM iterations per bin | 100 |
| `tol_em` | EM convergence tolerance | 1e-7 |
| `max_iter_perm` | Max permutation alignment iterations | 50 |
| `mask_threshold` | Threshold for dominant source detection | 0.65 (underdetermined) / 1.0 (determined) |

---

## 📊 Algorithm Pipeline

```
INPUT: Microphone Signals
    ↓
[1] STFT Analysis (compute_stft.m)
    ↓
[2] Feature Extraction (extract_complex_cosine_features.m)
    ↓
[3] Whitening (whiten_features.m)
    ↓
[4] EM Clustering (cluster_em_frequency_bins.m)
    ├─ Expectation Step (expectation.m)
    └─ Maximization Step (maximization.m)
    ↓
[5] Permutation Alignment (align_permutation.m)
    ├─ Global Optimization
    └─ Local Optimization
    ↓
[6] RTF Estimation (estimate_rtf.m)
    ↓
[7] Source Reconstruction (reconstruct_sources.m)
    ├─ Determined Case (N = M)
    └─ Underdetermined Case (N > M)
    ↓
[8] ISTFT Synthesis (compute_istft.m)
    ↓
[9] Evaluation (evaluate_separation.m)
    ↓
OUTPUT: Separated Source Signals
```

---

## 🧪 Sample Data

The repository includes two sample scenarios:

### **Determined Case** (N = M = 2)
- Files: `deter_mix_mic1.wav`, `deter_mix_mic2.wav`
- Number of sources: 2
- Number of microphones: 2
- Scenario: Well-determined mixing problem

### **Underdetermined Case** (N > M)
- Files: `under_mix_mic1.wav`, `under_mix_mic2.wav`
- Number of sources: 3 (or more)
- Number of microphones: 2
- Scenario: More sources than microphones (challenging)

For detailed information about the data, see `data/README.md`.

---

## 📖 How to Use Your Own Data

1. **Place your mixed audio files** in `data/sample_audio/`
2. **Update `main.m`** with your file paths:
   ```matlab
   [s1, FS] = audioread('data/sample_audio/your_mix_mic1.wav');
   [s2, FS] = audioread('data/sample_audio/your_mix_mic2.wav');
   ```
3. **Adjust parameters** (window size, number of sources, etc.)
4. **Run the script**: `main`

---

## 📈 Results & Evaluation

The script computes standard BSS metrics:

- **SDR** (Signal-to-Distortion Ratio) - dB
- **SIR** (Signal-to-Interference Ratio) - dB
- **SAR** (Signal-to-Artifact Ratio) - dB
- **ISR** (Image-to-Spatial Ratio) - dB

Results are printed to the console and can be saved for further analysis.

---

## 📚 Citation

If you use this code, please cite the original paper:

```bibtex
@article{bella2026underdetermined,
  title={Underdetermined convolutive blind source separation of speech signals using relaxed time--frequency sparsity and {RTF}-based recombination},
  author={Bella, Mostafa and Saylani, Hicham and Hosseini, Shahram},
  journal={Computer Speech \& Language},
  volume={101},
  pages={101992},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.csl.2026.101992}
}
```

---

## 📄 License

This code is provided for academic and research purposes. See the LICENSE file for details.

---

## 👨‍💻 Author

**Dr. Mostafa BELLA**

- 📧 [mbella@irap.omp.eu](mailto:mbella@irap.omp.eu)
- 💼 [LinkedIn](https://www.linkedin.com/in/mostafa-bella-2667a619/)
- 🔬 [ResearchGate](https://www.researchgate.net/profile/Mostafa-Bella)
- 📚 [Google Scholar](https://scholar.google.com/citations?user=zpKmY9IAAAAJ&hl=fr&oi=ao)

---

## 🙋 Support & Questions

For issues, questions, or suggestions, please:
1. Check the documentation in `data/README.md`
2. Review the comments in the source code
3. Open an issue on GitHub

---

*Last updated: August 2026*
