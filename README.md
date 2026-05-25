# 🧠 EEG Schizophrenia Classification

> Automated detection of Schizophrenia from EEG signals using Mel-Spectrograms and Deep Learning (EfficientNetB0 · CNN+BiLSTM)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF)](https://kaggle.com)
[![Paper](https://img.shields.io/badge/Reference-SCZ--SCAN%202023-green)](https://doi.org/10.1016/j.bspc.2023.105206)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [EEG → Spectrogram Pipeline (Full Detail)](#eeg--spectrogram-pipeline-full-detail)
  - [Step 1: Load Raw EEG](#step-1--load-raw-eeg-csv)
  - [Step 2: Z-Score Normalization](#step-2--z-score-normalization)
  - [Step 3: Sliding Window Segmentation](#step-3--sliding-window-segmentation)
  - [Step 4: Bipolar Montage Differences](#step-4--bipolar-montage-differences)
  - [Step 5: NaN Handling](#step-5--nan-handling)
  - [Step 6: Optional Wavelet Denoising](#step-6--optional-wavelet-denoising)
  - [Step 7: Mel Spectrogram (librosa)](#step-7--mel-spectrogram-librosa)
  - [Step 8: Log Power (dB) Conversion](#step-8--log-power-db-conversion)
  - [Step 9: Width Cropping](#step-9--width-cropping)
  - [Step 10: Standardisation](#step-10--standardisation)
  - [Step 11: Average 4 Pair Spectrograms](#step-11--average-4-pair-spectrograms)
  - [Step 12: Colourise & Compose 2×2 PNG](#step-12--colourise--compose-22-png)
- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Reference Paper](#reference-paper)

---

## Overview

This project implements a complete deep learning pipeline to classify EEG recordings as **Healthy Control (HC)** or **Schizophrenia (SZ)**:

1. Load raw multi-channel EEG CSVs (200 Hz, 16 electrodes)
2. Apply **Bipolar Double-Banana montage** (LL, LP, RP, RR chains)
3. Segment using **4-second sliding windows, 75% overlap**
4. Per bipolar pair: compute **Mel-spectrogram → dB → standardise**
5. Average 4 pair spectrograms per montage → **128×256 panel**
6. Compose **2×2 viridis-coloured PNG** (one per window)
7. Train **EfficientNetB0** or **CNN+BiLSTM** for binary classification

---

## Dataset

**Source:** [EEG data from basic sensory task in Schizophrenia](https://www.kaggle.com/datasets/broach/button-tone-sz) — PhysioNet / Kaggle

| Property | Value |
|---|---|
| Total Subjects | 81 (32 HC + 49 SZ) |
| CSV files available | HC: subjects 1–24, 66 (25 files) · SZ: subjects 67–81 (15 files) |
| Missing from upload | Subjects 25–65 (34 SZ subjects have no CSV) |
| Sampling Rate | 200 Hz |
| Channels | 16 EEG electrodes (Fp1, F7, T3, T5, O1, Fp2, F8, T4, T6, O2, F3, C3, P3, F4, C4, P4) |
| Task | Auditory button-tone sensory task |
| File format | CSV — columns: `trial, sample, id, time, [16 electrodes]` |

---

## Project Structure

```
minor_project/
├── generate_synthetic_data.py   # Generates simulated mock EEG dataset for testing [NEW]
├── run_pipeline.py              # Master orchestrator script to run pipeline end-to-end [NEW]
├── generate_spectrograms.py     # EEG → Mel-spectrogram image pipeline  (v4)
├── train_efficientnet.py        # EfficientNetB0 training & evaluation (cached features support)
├── train_cnn_bilstm.py          # Hybrid CNN + BiLSTM training (cached sequence features support)
├── train_hybrid_model.py        # RF, SVM & Ensemble experiments (caching support)
├── train_hybrid_svm.py          # SVM baseline experiments (caching support)
├── make_ppt.py                  # Auto-generate project PowerPoint
├── generate_report.py           # PDF report generator
├── how-to-make-spectrogram-from-eeg.ipynb   # Reference notebook (cdeotte)
├── 1-s2.0-S1746809423006390-main.pdf        # SCZ-SCAN reference paper
├── .gitignore                   # Ignore list for Git tracking [NEW]
└── README.md                    # This file

# Generated on Kaggle or locally at runtime
/kaggle/working/ or minor_project/
├── dummy_dataset/              # Simulated EEG dataset
├── eeg_images/                 # Generated spectrogram panel PNGs
│   ├── HC/   subj001_t0001_w000000.png  ...
│   └── SZ/   subj067_t0001_w000000.png  ...
├── features_cache/             # Saved feature arrays (.npy files)
│   ├── train_features_1280.npy
│   ├── train_features_seq.npy
│   └── ...
└── model_output/               # Trained models, plots, and logs
    ├── best_phase1.keras / best_phase2.keras
    ├── efficientnet_eeg_final.keras
    ├── confusion_matrix.png
    ├── confusion_matrix_report.txt
    ├── roc_curve.png
    └── curves_phase1.png / curves_phase2.png
```

---

## EEG → Spectrogram Pipeline (Full Detail)

This section documents **every formula and transformation** used to convert raw EEG signals into spectrogram images.  
The implementation follows the reference notebook [`how-to-make-spectrogram-from-eeg.ipynb`](how-to-make-spectrogram-from-eeg.ipynb) by cdeotte exactly.

---

### Step 1 — Load Raw EEG CSV

Each subject CSV has the layout:

```
trial | sample | id | time | Fp1 | F7 | T3 | T5 | O1 | Fp2 | F8 | T4 | T6 | O2 | F3 | C3 | P3 | F4 | C4 | P4
```

The 16 electrode columns are extracted:

```python
ELECTRODE_NAMES = ["Fp1","F7","T3","T5","O1",
                   "Fp2","F8","T4","T6","O2",
                   "F3","C3","P3","F4","C4","P4"]

trial_values = trial_df[ELECTRODE_NAMES].values   # shape: (N_samples, 16)
```

---

### Step 2 — Z-Score Normalization

**Per-trial, per-channel** normalization (Paper Eq. 1):

$$x_{norm} = \frac{x - \mu_{ch}}{\sigma_{ch} + \varepsilon}$$

Where:
- $\mu_{ch}$ = mean across all time samples for channel `ch`
- $\sigma_{ch}$ = standard deviation across all time samples for channel `ch`
- $\varepsilon = 10^{-8}$ (numerical stability)

```python
mean = trial_values.mean(axis=0)          # shape: (16,)
std  = trial_values.std(axis=0) + 1e-8   # shape: (16,)
trial_values = (trial_values - mean) / std
```

> **Why?** Removes DC offset and amplitude differences between electrodes/subjects.
> Applied once per trial before windowing.

---

### Step 3 — Sliding Window Segmentation

**Paper Section 3.2:** 4-second windows with 75% overlap.

| Parameter | Formula | Value |
|---|---|---|
| Window size | `4 sec × 200 Hz` | **800 samples** |
| Stride | `25% × 800` | **200 samples** |
| Overlap | `1 - stride/window` | **75%** |
| Min trial length | ≥ window_size | 800 samples |

```python
WINDOW_SIZE = 800   # 4 s × 200 Hz
STRIDE_SIZE = 200   # 25% of window = 75% overlap

for win_start in range(0, n - WINDOW_SIZE + 1, STRIDE_SIZE):
    window = trial_values[win_start : win_start + WINDOW_SIZE]
    # window shape: (800, 16)
```

Number of windows per trial of length N:

$$n_{windows} = \left\lfloor \frac{N - W}{S} \right\rfloor + 1$$

Where $W = 800$, $S = 200$.

---

### Step 4 — Bipolar Montage Differences

Four montage chains are defined. Each chain uses 5 electrodes → **4 consecutive bipolar differences**:

| Chain | Electrodes | Bipolar pairs |
|---|---|---|
| **LL** (Left Lateral) | Fp1, F7, T3, T5, O1 | Fp1−F7, F7−T3, T3−T5, T5−O1 |
| **LP** (Left Parasagittal) | Fp1, F3, C3, P3, O1 | Fp1−F3, F3−C3, C3−P3, P3−O1 |
| **RP** (Right Parasagittal) | Fp2, F4, C4, P4, O2 | Fp2−F4, F4−C4, C4−P4, P4−O2 |
| **RR** (Right Lateral) | Fp2, F8, T4, T6, O2 | Fp2−F8, F8−T4, T4−T6, T6−O2 |

**Formula for each bipolar pair signal:**

$$x_{kk} = \text{EEG}[\text{col}_{kk}] - \text{EEG}[\text{col}_{kk+1}], \quad kk \in \{0,1,2,3\}$$

```python
MONTAGE_FEATS = [
    ["Fp1","F7","T3","T5","O1"],   # LL
    ["Fp1","F3","C3","P3","O1"],   # LP
    ["Fp2","F4","C4","P4","O2"],   # RP
    ["Fp2","F8","T4","T6","O2"],   # RR
]

x = window[:, col_idx[kk]] - window[:, col_idx[kk+1]]
# x shape: (800,)  — one bipolar signal
```

> **Why bipolar?** Bipolar referencing cancels common noise (EMG, mains) shared between adjacent electrodes, revealing local brain activity.

> **Why average spectrograms instead of averaging signals?**
> Averaging signals first:  
> `spec( (Fp1−F7 + F7−T3 + T3−T5 + T5−O1) / 4 ) = spec( (Fp1−O1) / 4 )`  
> → collapses to just two electrodes, losing information!
>
> Averaging spectrograms (correct formula):  
> `( spec(Fp1−F7) + spec(F7−T3) + spec(T3−T5) + spec(T5−O1) ) / 4`  
> → preserves all 5 electrode signals since spectrogram is a **non-linear** operation.

---

### Step 5 — NaN Handling

Before computing the spectrogram, NaN values in each bipolar signal are filled:

```python
m = np.nanmean(x)

if np.isnan(x).mean() < 1:       # at least one non-NaN value exists
    x = np.nan_to_num(x, nan=m)  # replace NaNs with channel mean
else:
    x[:] = 0                     # all-NaN channel → fill with zeros
```

---

### Step 6 — Optional Wavelet Denoising

Optional hard-threshold wavelet denoising (disabled by default; enable with `--wavelet db8`).

**Algorithm** (Donoho & Johnstone universal threshold):

1. Decompose signal using DWT:  
   `coeff = pywt.wavedec(x, wavelet, mode="per")`

2. Estimate noise standard deviation from finest-scale coefficients:  
   $$\hat{\sigma} = \frac{1}{0.6745} \cdot \text{MAD}(c_{-1})$$  
   where $\text{MAD}(d) = \mathbb{E}[|d - \mathbb{E}[d]|]$

3. Compute universal threshold:  
   $$\lambda = \hat{\sigma} \cdot \sqrt{2 \ln N}$$

4. Hard-threshold all detail coefficients:  
   $$c_j^* = c_j \cdot \mathbf{1}[|c_j| > \lambda]$$

5. Reconstruct: `x_clean = pywt.waverec(coeff*, wavelet, mode="per")`

```python
def denoise_signal(x, wavelet="db8", level=1):
    coeff   = pywt.wavedec(x, wavelet, mode="per")
    sigma   = (1 / 0.6745) * np.mean(np.abs(coeff[-level] - np.mean(coeff[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = [pywt.threshold(c, value=uthresh, mode="hard") for c in coeff[1:]]
    return pywt.waverec(coeff, wavelet, mode="per")[:len(x)]
```

---

### Step 7 — Mel Spectrogram (librosa)

**Exact notebook parameters** (from `how-to-make-spectrogram-from-eeg.ipynb`):

```python
mel_spec = librosa.feature.melspectrogram(
    y          = x,           # bipolar signal, shape: (800,)
    sr         = 200,         # sampling rate: 200 Hz
    hop_length = len(x)//256, # = 800//256 = 3 samples → 256 time frames
    n_fft      = 1024,        # FFT window size
    n_mels     = 128,         # number of Mel filter banks (image height)
    fmin       = 0,           # lowest frequency: 0 Hz
    fmax       = 20,          # highest frequency: 20 Hz
    win_length = 128,         # analysis window: 128 samples = 0.64 s
)
# mel_spec shape: (128, ~256)
```

**Parameter explanation:**

| Parameter | Value | Meaning |
|---|---|---|
| `sr` | 200 Hz | EEG sampled at 200 times/second |
| `hop_length` | `len(x) // 256` = 3 | Step between frames → ensures exactly 256 time columns |
| `n_fft` | 1024 | FFT points → frequency resolution = 200/1024 ≈ 0.195 Hz/bin |
| `n_mels` | 128 | Output image height = 128 pixels |
| `fmin` | 0 Hz | Include DC component |
| `fmax` | 20 Hz | EEG delta(0.5–4), theta(4–8), alpha(8–13), beta(13–20) Hz |
| `win_length` | 128 | Hann window of 128 samples = 0.64 s |

**Mel filterbank formula:**

Mel scale converts linear frequency $f$ (Hz) to perceptual scale:

$$m = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right)$$

The 128 Mel filter centres are spaced linearly in Mel space between $f_{min}=0$ and $f_{max}=20$ Hz.

**Power spectrogram:**

$$S(m, t) = \left| \sum_{n=0}^{N-1} x[n + t \cdot H] \cdot w[n] \cdot e^{-j2\pi kn/N} \right|^2$$

Where $N=1024$ (n_fft), $H=3$ (hop_length), $w[n]$ = Hann window of length 128 (zero-padded to N).

> **Why N_FFT=1024?**  
> At 200 Hz, frequency resolution = 200/1024 ≈ 0.2 Hz/bin.  
> The 0–20 Hz band contains ~103 unique FFT bins, more than enough to fill 128 Mel bands.  
> Using N_FFT=256 → only ~26 FFT bins in 0–20 Hz → severe horizontal stripe artifacts.

---

### Step 8 — Log Power (dB) Conversion

Convert power spectrogram to decibels, referenced to the **maximum power in that spectrogram**:

$$S_{dB}(m, t) = 10 \cdot \log_{10}\!\left(\frac{S(m,t)}{\max_{m,t} S(m,t)}\right)$$

```python
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
# mel_spec_db shape: (128, frames)
# values typically in range [-80, 0] dB
```

> **Why `ref=np.max`?**  
> Normalises each spectrogram to its own maximum so the colour scale is relative to the strongest frequency in that window, making images comparable across subjects and trials.

---

### Step 9 — Width Cropping

Crop the time axis to the nearest multiple of 32 ≤ 256 (notebook-exact):

```python
width = (mel_spec_db.shape[1] // 32) * 32   # e.g. 267 → 256
width = min(width, 256)                       # cap at IMG_W

mel_spec_db = mel_spec_db[:, :width]         # shape: (128, width)

# Pad if width < 256 to keep accumulator shape consistent
if width < 256:
    mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, 256-width)))
```

---

### Step 10 — Standardisation

Map dB values (typically −80…0) to approximately [−1, 1]:

$$S_{std}(m,t) = \frac{S_{dB}(m,t) + 40}{40}$$

```python
mel_spec_db = (mel_spec_db + 40.0) / 40.0
```

| dB value | Standardised value |
|---|---|
| 0 dB | +1.0 |
| −40 dB | 0.0 |
| −80 dB | −1.0 |

> **Why +40/40?**  
> The typical dynamic range of a dB-referenced spectrogram is −80…0 dB.  
> Shifting by +40 and dividing by 40 centres the values around 0 and normalises to unit scale, which stabilises gradient updates during training.

---

### Step 11 — Average 4 Pair Spectrograms

For each montage chain $k$ (LL, LP, RP, RR), accumulate all 4 bipolar-pair spectrograms and average:

$$\text{img}[:,:,k] = \frac{1}{4} \sum_{kk=0}^{3} S_{std}^{(kk)}$$

```python
img = np.zeros((128, 256, 4), dtype=np.float32)  # 4-channel accumulator

for k, col_idx in enumerate(MONTAGE_IDX):
    for kk in range(4):
        x = window[:, col_idx[kk]] - window[:, col_idx[kk+1]]
        # ... NaN fill, optional denoise ...
        mel_spec = librosa.feature.melspectrogram(y=x, ...)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)[:, :width]
        mel_spec_db = (mel_spec_db + 40) / 40
        img[:, :, k] += mel_spec_db

    img[:, :, k] /= 4.0   # average over 4 pairs
```

**Result:** `img` shape = `(128, 256, 4)` — one 128×256 float32 panel per montage.

---

### Step 12 — Colourise & Compose 2×2 PNG

Each of the 4 montage panels is colourised using the **viridis** colourmap, then arranged into a 2×2 grid:

```
┌──────────────────┬──────────────────┐
│   LL  (128×256)  │   LP  (128×256)  │
├──────────────────┼──────────────────┤
│   RP  (128×256)  │   RR  (128×256)  │
└──────────────────┴──────────────────┘
   Final PNG size: 256 × 512 × 3  (H × W × RGB)
```

**Colourisation via viridis LUT:**

```python
# Pre-build viridis LUT once (256 × 3 uint8)
_VIRIDIS_LUT = (plt.cm.viridis(np.linspace(0,1,256))[:,:3] * 255).astype(np.uint8)

def apply_viridis(arr2d):
    mn, mx = arr2d.min(), arr2d.max()
    if mx - mn > 0:
        idx = ((arr2d - mn) / (mx - mn) * 255).clip(0,255).astype(np.uint8)
    else:
        idx = np.zeros_like(arr2d, dtype=np.uint8)
    return _VIRIDIS_LUT[idx]   # shape: (128, 256, 3)

panels = [apply_viridis(img[:,:,k]) for k in range(4)]
grid = np.vstack([np.hstack([panels[0], panels[1]]),
                  np.hstack([panels[2], panels[3]])])  # (256, 512, 3)

Image.fromarray(grid).save(save_path)
```

**Complete per-window summary:**

```
Window (800, 16)
    │
    ├─ [for each of 4 montages k]
    │       ├─ [for each of 4 bipolar pairs kk]
    │       │       x = EEG[col_kk] - EEG[col_kk+1]      # (800,)
    │       │       x = nan_to_num(x, mean)
    │       │       [optional: wavelet denoise]
    │       │       mel  = melspectrogram(x, hop=3, ...)   # (128, 267)
    │       │       db   = power_to_db(mel, ref=max)       # (128, 267)
    │       │       db   = db[:, :256]                     # (128, 256)
    │       │       db   = (db + 40) / 40                  # standardise
    │       │       img[:,:,k] += db
    │       └─ img[:,:,k] /= 4.0
    │
    └─ grid = 2×2 viridis-coloured panels → PNG (256×512×3)
```

---

## Model Architectures

### Model 1 — EfficientNetB0

```
Input (128 × 256 × 3)
    │
EfficientNetB0 backbone (ImageNet pretrained, 237 layers)
    │  output: (4, 8, 1280)
GlobalAveragePooling2D → (1280,)
    │
BatchNormalization
    │
Dropout(0.4) → Dense(256, ReLU) → Dropout(0.3)
    │
Dense(1, Sigmoid) → P(SZ)
```

**Two-phase training:**

| Phase | Backbone | LR | Epochs |
|---|---|---|---|
| 1 — Feature Extraction | Frozen | 1e-4 | 30 |
| 2 — Fine-Tuning | Top-20 layers unfrozen | 1e-5 | 15 |

---

### Model 2 — Hybrid CNN + Bidirectional LSTM

```
Input (128 × 256 × 3)
    │
SpecAugment (time mask ±30, freq mask ±20)
    │
EfficientNetB0 backbone → (4, 8, 1280)
    │
Mean-pool over frequency axis (axis=1) → (8, 1280)   ← 8 time steps
    │
BiLSTM(256, return_sequences=True) → (8, 512)
    │
BiLSTM(128) → (256,)
    │
BatchNorm → Dropout(0.4) → Dense(256, ReLU) → Dropout(0.3)
    │
Dense(1, Sigmoid) → P(SZ)
```

**Why BiLSTM?** The x-axis of the spectrogram is time (256 bins → 8 pooled steps after CNN). BiLSTM captures how frequency patterns **evolve** left→right and right→left, modelling EEG temporal dynamics.

**Extra features:**

| Feature | Detail |
|---|---|
| Class weights | `compute_class_weight("balanced")` — fixes HC/SZ subject imbalance |
| SpecAugment | Random time + frequency masking during training |
| LR schedule | Cosine Decay Restarts |
| Data pipeline | `tf.data` with `prefetch(AUTOTUNE)` |

---

## Installation

```bash
pip install librosa PyWavelets tqdm Pillow tensorflow python-pptx fpdf2
```

---

## Usage

### 🚀 Quickstart: End-to-End Demo (Local Synthetic Run)

You can verify and run the entire pipeline end-to-end locally with a mock dataset using:
```bash
python run_pipeline.py
```
This script will automatically:
1. Generate a small synthetic EEG dataset under `dummy_dataset/`.
2. Generate Mel-spectrogram panel PNGs in `eeg_images/`.
3. Pre-extract features from images using EfficientNetB0 and cache them in `features_cache/`.
4. Train `train_efficientnet.py` and `train_cnn_bilstm.py` on the cached features for 2 epochs.
5. Train scikit-learn models (`train_hybrid_model.py` and `train_hybrid_svm.py`) directly on the shared cached features.
6. Auto-generate the final presentation (`EEG_Schizophrenia_Detection.pptx`) and PDF report (`minor_project_report.pdf`).

---

### Generate Spectrogram Images (Kaggle)

```python
# Cell 1 — Install
!pip install -q librosa PyWavelets tqdm Pillow

# Cell 2 — Generate 4000 spectrogram images (HC-2000, SZ-2000)
!python /kaggle/working/generate_spectrograms.py \
    --dataset /kaggle/input/datasets/broach/button-tone-sz \
    --out     /kaggle/working/eeg_images \
    --images_per_class 2000 \
    --workers 4

# Optional: enable wavelet denoising
# --wavelet db8
```

### Train EfficientNetB0

```bash
python train_efficientnet.py \
  --data      ./eeg_images \
  --out       ./model_output \
  --epochs    30 \
  --batch     32 \
  --fine_tune 10
```

### Train CNN + BiLSTM

```bash
python train_cnn_bilstm.py \
  --data      ./eeg_images \
  --out       ./cnn_bilstm_output \
  --epochs    40 \
  --batch     16 \
  --fine_tune 20
```

> Use `--batch 16` for BiLSTM — it uses more GPU memory than plain CNN.

### Generate PowerPoint

```bash
python make_ppt.py
# Output: EEG_Schizophrenia_Detection.pptx
```

---

## Results

Reference results from the SCZ-SCAN paper on this dataset:

| Method | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| SCZ-SCAN (paper) | **96%** | 96% | 95% | — |
| EfficientNetB0 (this project) | TBD | TBD | TBD | TBD |
| CNN + BiLSTM (this project) | TBD | TBD | TBD | TBD |

**Evaluation outputs saved per run:**

| File | Contents |
|---|---|
| `confusion_matrix.png` | Raw counts + row-normalised side-by-side |
| `confusion_matrix_report.txt` | TP, TN, FP, FN, sensitivity, specificity, F1 |
| `roc_curve.png` | FPR vs TPR curve with AUC |
| `curves_phase1/2.png` | Train/val accuracy & loss per epoch |

---

## Reference Paper

> G. Sahu et al., **"SCZ-SCAN: An automated Schizophrenia detection model using EEG signals"**,
> *Biomedical Signal Processing and Control*, 86 (2023) 105206.
> DOI: [10.1016/j.bspc.2023.105206](https://doi.org/10.1016/j.bspc.2023.105206)

Methodology adopted from the paper:

| Element | Adopted |
|---|---|
| Z-score normalization (Eq. 1) | ✅ |
| 4-second windows, 75% overlap (Section 3.2) | ✅ |
| Dataset-2 (button-tone-sz) | ✅ |
| Bipolar double-banana montage (LL/LP/RP/RR) | ✅ |
| Mel-spectrograms + EfficientNetB0 | ✅ (this project's approach) |
| CNN+BiLSTM temporal modelling | ✅ (this project's extension) |
| CWT scalograms + SCZ-SCAN CNN | ➡️ (paper's approach — not used here) |
