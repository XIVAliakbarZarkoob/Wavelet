# Wavelet Transformations: 1D and 2D Analysis

## Overview

This repository focuses on the implementation and analysis of wavelet transforms for both one-dimensional (1D) and two-dimensional (2D) data. Wavelet analysis is a powerful tool for signal processing, data compression, and feature extraction, offering time-frequency localization and adaptability to non-stationary signals. 

The repository includes implementations and results for various wavelets applied to simulated signals and images, enabling tasks such as noise reduction and cycle slip detection.

---

## Features

### Wavelet Types:
1. **Haar Wavelet**  
   - Simple and efficient for basic signal processing tasks.  
2. **Daubechies (D4, D6)**  
   - Advanced wavelets offering better accuracy and multi-level decomposition.  
3. **Mexican Hat Wavelet**  
   - Ideal for edge detection and singularity analysis.  
4. **Symlet Wavelet (S2)**  
   - A symmetric variant of Daubechies wavelets for improved signal reconstruction.

### Applications:
- **Noise Reduction**  
  Using wavelets to filter out noise from 1D signals while retaining essential features.
  
- **Cycle Slip Detection in GNSS Signals**  
  Applying wavelets to identify and detect discontinuities in GPS signals.

- **2D Image Denoising**  
  Enhancing image quality by removing Salt & Pepper and Gaussian noise.

## Results

### 1D Signal Processing:
- Noise reduction results demonstrate significant improvements with Daubechies6 Wavelet at level 2 decomposition, yielding the least error in signal reconstruction.

### 2D Image Processing:
- Visualizations of denoised images highlight the effectiveness of different wavelets, with comparative performance metrics available in the results section.

