# Sample Audio Data

This directory contains sample audio mixtures for testing the Blind Source Separation (BSS) algorithm.

## Files

### Determined Case (N = M = 2)
- **deter_mix_mic1.wav** - Mixed signal on microphone 1 (determined case)
- **deter_mix_mic2.wav** - Mixed signal on microphone 2 (determined case)

**Parameters:**
- Number of sources: 2
- Number of microphones: 2
- Status: Well-determined problem (N = M)

### Underdetermined Case (N > M)
- **under_mix_mic1.wav** - Mixed signal on microphone 1 (underdetermined case)
- **under_mix_mic2.wav** - Mixed signal on microphone 2 (underdetermined case)

**Parameters:**
- Number of sources: 3 or more
- Number of microphones: 2
- Status: Underdetermined problem (N > M, challenging!)

---

## Usage

To run the algorithm on these sample data:

1. Open `main.m` and set:
   ```matlab
   params.num_sources = 2;  % for determined case
   % or
   params.num_sources = 3;  % for underdetermined case
   ```

2. Run:
   ```matlab
   main
   ```

3. Separated signals will be saved to `../results/` directory

---

## Notes

- All audio files are in WAV format (16-bit PCM)
- Sampling rate: 16 kHz (typically)
- Signal length: 160,000 samples (10 seconds at 16 kHz)

---

*Last updated: August 2026*
