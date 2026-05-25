"""
generate_spectrograms.py  (v4 — exact notebook-aligned)
========================================================
Generates EEG Mel-spectrogram images (HC & SZ classes).

Image generation logic copied directly from:
  how-to-make-spectrogram-from-eeg.ipynb  (cdeotte / HMS-HBAC)

Formula (notebook-exact)
------------------------
For each of the 4 montage chains (LL, LP, RP, RR):
  1. Compute 4 bipolar-difference signals  (consecutive electrode pairs)
  2. For EACH pair:
       a. Fill NaNs with channel mean (or 0 if all-NaN)
       b. Optionally wavelet-denoise
       c. mel_spec = librosa.feature.melspectrogram(
              y=x, sr=200, hop_length=len(x)//256,
              n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)
       d. width  = (mel_spec.shape[1] // 32) * 32   # crop to nearest ×32
       e. mel_db = librosa.power_to_db(mel_spec, ref=np.max)[:, :width]
       f. mel_db = (mel_db + 40) / 40               # standardise to ~[-1,1]
  3. Accumulate and divide by 4  →  one 128×256 panel per chain

Output: 4-channel float32 numpy array (128 × 256 × 4) stored as PNG
        using viridis colourmap in a 2×2 grid — identical to the
        notebook's displayed spectrograms.

Dataset layout expected
-----------------------
  <dataset_root>/
      demographic.csv
      <subj>.csv/
          <subj>.csv        ← Kaggle nested layout
      OR
      <subj>.csv            ← flat layout

Usage
-----
  python generate_spectrograms.py            # Kaggle defaults
  python generate_spectrograms.py --dataset /path/to/button-tone-sz --out ./out

Dependencies
------------
  pip install librosa PyWavelets tqdm Pillow
  (numpy, pandas, matplotlib pre-installed on Kaggle)
"""

import os
import argparse
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import librosa
from tqdm import tqdm

# PyWavelets is optional
try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False
    print("[WARN] PyWavelets not found – wavelet denoising disabled. "
          "Install with: pip install PyWavelets")

warnings.filterwarnings("ignore")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate EEG Mel-spectrogram images (HC-2000, SZ-2000)"
    )
    p.add_argument("--dataset",
                   default="/kaggle/input/datasets/broach/button-tone-sz",
                   help="Path to button-tone-sz root directory")
    p.add_argument("--out",
                   default="/kaggle/working/eeg_images",
                   help="Output directory")
    p.add_argument("--images_per_class", type=int, default=2000)
    p.add_argument("--wavelet",          type=str,  default=None,
                   help="Wavelet for optional denoising (e.g. db8)")
    p.add_argument("--workers",          type=int,
                   default=min(4, os.cpu_count() or 2),
                   help="Parallel workers (default: cpu_count, max 4)")
    # Ignore Jupyter's own -f kernel.json argument
    args, _ = p.parse_known_args()
    return args


# ── EEG / Montage constants ───────────────────────────────────────────────────
CHANNEL_NAMES = [
    "trial", "sample", "id", "time",
    "Fp1", "F7",  "T3", "T5", "O1",
    "Fp2", "F8",  "T4", "T6", "O2",
    "F3",  "C3",  "P3",
    "F4",  "C4",  "P4",
]
ELECTRODE_NAMES = CHANNEL_NAMES[4:]   # 16 electrode columns

MONTAGE_NAMES = ["LL", "LP", "RP", "RR"]
MONTAGE_FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],   # LL
    ["Fp1", "F3", "C3", "P3", "O1"],   # LP
    ["Fp2", "F4", "C4", "P4", "O2"],   # RP
    ["Fp2", "F8", "T4", "T6", "O2"],   # RR
]

IMG_H      = 128   # mel bins  (height of each panel)  — matches n_mels
IMG_W      = 256   # time bins (width of each panel)   — matches hop_length divisor
SR         = 200   # Hz  (EEG sampling rate)
N_FFT      = 1024  # notebook: n_fft=1024
N_MELS     = 128   # notebook: n_mels=128
FMIN       = 0     # notebook: fmin=0
FMAX       = 20    # notebook: fmax=20
WIN_LENGTH = 128   # notebook: win_length=128

# ── Windowing (from paper Section 3.2) ───────────────────────────────────────
# Paper: 4-second window, 75% overlap → stride = 25% × 800 = 200 samples
WINDOW_SIZE   = 800   # 4 s × 200 Hz
STRIDE_SIZE   = 200   # 75% overlap
MIN_TRIAL_LEN = WINDOW_SIZE

# Pre-build viridis LUT (256 × 3 uint8) for fast colourisation
_VIRIDIS_LUT = (plt.cm.viridis(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

# Pre-build electrode → column index map once
_ELEC_IDX    = {name: i for i, name in enumerate(ELECTRODE_NAMES)}
_MONTAGE_IDX = [[_ELEC_IDX[c] for c in cols] for cols in MONTAGE_FEATS]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _apply_viridis(arr2d: np.ndarray) -> np.ndarray:
    """Map a 2-D float array → H×W×3 uint8 RGB via viridis LUT."""
    mn, mx = arr2d.min(), arr2d.max()
    span = mx - mn
    if span > 0:
        idx = ((arr2d - mn) / span * 255).clip(0, 255).astype(np.uint8)
    else:
        idx = np.zeros_like(arr2d, dtype=np.uint8)
    return _VIRIDIS_LUT[idx]


def _maddest(d):
    return np.mean(np.abs(d - np.mean(d)))


def denoise_signal(x: np.ndarray, wavelet: str = "db8", level: int = 1) -> np.ndarray:
    """Wavelet hard-threshold denoising. Returns x unchanged if PyWavelets is missing."""
    if not _PYWT_AVAILABLE:
        return x
    try:
        coeff   = pywt.wavedec(x, wavelet, mode="per")
        sigma   = (1 / 0.6745) * _maddest(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = [pywt.threshold(c, value=uthresh, mode="hard") for c in coeff[1:]]
        return pywt.waverec(coeff, wavelet, mode="per")[:len(x)]
    except Exception:
        return x


# ── Core: one window → one PNG ────────────────────────────────────────────────
# This function is a direct port of spectrogram_from_eeg() from the notebook:
#   how-to-make-spectrogram-from-eeg.ipynb
def trial_to_spectrogram_image(trial_values: np.ndarray,
                                save_path: str,
                                use_wavelet=None) -> bool:
    """
    trial_values : numpy float32 array shape (WINDOW_SIZE, 16) – electrode columns.

    Exact notebook logic (spectrogram_from_eeg)
    --------------------------------------------
    img = np.zeros((128, 256, 4), dtype='float32')   ← 4-channel accumulator

    For each of the 4 montage chains k (LL, LP, RP, RR):
      For each of the 4 consecutive bipolar pairs kk:
        1. x = eeg[COLS[kk]] - eeg[COLS[kk+1]]
        2. Fill NaNs: if mean<1 nan → nan_to_num(x, nan=m) else x[:]=0
        3. Optional wavelet denoise
        4. mel_spec = librosa.feature.melspectrogram(
               y=x, sr=200, hop_length=len(x)//256,
               n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)
        5. width = (mel_spec.shape[1]//32)*32        ← crop to nearest ×32
        6. mel_db = power_to_db(mel_spec, ref=max)[:, :width].astype(float32)
        7. mel_db = (mel_db + 40) / 40               ← standardise
        8. img[:,:,k] += mel_db
      img[:,:,k] /= 4.0                              ← average the 4 pairs

    The 4 panels are then colourised with viridis and composed into a 2×2 PNG.
    Returns True on success.
    """
    n_samples = trial_values.shape[0]

    # 4-channel float32 accumulator — matches notebook exactly
    img = np.zeros((IMG_H, IMG_W, 4), dtype=np.float32)

    for k, col_idx in enumerate(_MONTAGE_IDX):
        for kk in range(4):   # 4 consecutive bipolar pairs per chain

            # ── 1. Bipolar difference (notebook: eeg[COLS[kk]] - eeg[COLS[kk+1]]) ──
            x = (trial_values[:, col_idx[kk]] -
                 trial_values[:, col_idx[kk + 1]]).astype(np.float32)

            # ── 2. Fill NaNs (notebook: nan_to_num or zeros) ─────────────────────
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            # ── 3. Optional wavelet denoising ─────────────────────────────────────
            if use_wavelet:
                x = denoise_signal(x, wavelet=use_wavelet)

            # ── 4. Raw spectrogram (notebook: hop_length=len(x)//256) ─────────────
            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=SR,
                hop_length=len(x) // IMG_W,   # ← exact notebook formula
                n_fft=N_FFT,
                n_mels=N_MELS,
                fmin=FMIN,
                fmax=FMAX,
                win_length=WIN_LENGTH,
            )

            # ── 5. Crop width to nearest multiple of 32 (notebook exact) ──────────
            width = (mel_spec.shape[1] // 32) * 32   # e.g. 256 → 256
            width = min(width, IMG_W)                 # cap at IMG_W

            # ── 6. Log-power dB (notebook: power_to_db ref=max) ──────────────────
            mel_spec_db = librosa.power_to_db(
                mel_spec, ref=np.max
            ).astype(np.float32)[:, :width]           # shape: (128, width)

            # Pad if width < IMG_W so accumulator shapes always match
            if width < IMG_W:
                mel_spec_db = np.pad(
                    mel_spec_db, ((0, 0), (0, IMG_W - width))
                )

            # ── 7. Standardise to ~[-1, 1] (notebook: (mel_db+40)/40) ────────────
            mel_spec_db = (mel_spec_db + 40.0) / 40.0

            # ── 8. Accumulate into channel k ──────────────────────────────────────
            img[:, :, k] += mel_spec_db

        # Average the 4 bipolar-pair spectrograms (notebook: img[:,:,k] /= 4.0)
        img[:, :, k] /= 4.0

    # ── Colourise each channel with viridis and compose 2×2 PNG ─────────────
    panels = [_apply_viridis(img[:, :, k]) for k in range(4)]
    grid = np.vstack([
        np.hstack([panels[0], panels[1]]),
        np.hstack([panels[2], panels[3]]),
    ])   # shape: (2*IMG_H, 2*IMG_W, 3)

    Image.fromarray(grid).save(save_path)
    return True


# ── CSV path helper ───────────────────────────────────────────────────────────
def get_csv_path(dataset_root: Path, subj: int):
    nested = dataset_root / f"{subj}.csv" / f"{subj}.csv"
    if nested.exists():
        return str(nested)
    flat = dataset_root / f"{subj}.csv"
    if flat.exists():
        return str(flat)
    return None


# ── Multiprocessing worker ────────────────────────────────────────────────────
def _worker(task):
    """task = (trial_values_np, save_path, use_wavelet)"""
    trial_values, save_path, use_wavelet = task
    try:
        return trial_to_spectrogram_image(trial_values, save_path, use_wavelet)
    except Exception:
        return False


# ── Subject-level generator ───────────────────────────────────────────────────
def generate_images_for_subjects(
    dataset_root: Path,
    subject_list: list,
    label_str: str,
    out_dir: Path,
    max_images: int,
    use_wavelet=None,
    num_workers: int = 2,
) -> int:
    save_dir = out_dir / label_str
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: collect tasks (read each CSV once, use groupby) ─────────────
    tasks = []
    print(f"\n[{label_str}] Scanning subjects …")

    for subj in tqdm(subject_list, desc=f"  Loading {label_str} CSVs"):
        if len(tasks) >= max_images:
            break
        csv_path = get_csv_path(dataset_root, subj)
        if csv_path is None:
            print(f"  [WARN] Subject {subj}: CSV not found — skipping")
            continue
        try:
            df = pd.read_csv(csv_path, header=None)
        except Exception as exc:
            print(f"  [ERROR] Subject {subj}: {exc}")
            continue

        # Assign column names
        n_extra = df.shape[1] - len(CHANNEL_NAMES)
        df.columns = CHANNEL_NAMES + [f"extra_{i}" for i in range(n_extra)]

        # Check all electrode columns present
        if not all(c in df.columns for c in ELECTRODE_NAMES):
            print(f"  [WARN] Subject {subj}: missing electrode columns — skipping")
            continue

        # groupby trial → apply sliding windows (paper Section 3.2)
        for trial_num, trial_df in df.groupby("trial"):
            if len(tasks) >= max_images:
                break
            if len(trial_df) < MIN_TRIAL_LEN:
                continue

            # Extract electrode data and z-score normalise (paper Eq. 1)
            trial_values = trial_df[ELECTRODE_NAMES].values.astype(np.float32)
            mean = trial_values.mean(axis=0)
            std  = trial_values.std(axis=0) + 1e-8
            trial_values = (trial_values - mean) / std

            # Sliding windows: 4 s window, 75% overlap (stride = 25%)
            n = len(trial_values)
            for win_start in range(0, n - WINDOW_SIZE + 1, STRIDE_SIZE):
                if len(tasks) >= max_images:
                    break
                window = trial_values[win_start: win_start + WINDOW_SIZE]
                save_path = str(
                    save_dir /
                    f"subj{subj:03d}_t{int(trial_num):04d}_w{win_start:06d}.png"
                )
                tasks.append((window, save_path, use_wavelet))

    print(f"[{label_str}] {len(tasks)} trials queued  "
          f"({num_workers} process{'es' if num_workers > 1 else ''})")

    # ── Phase 2: parallel image generation ───────────────────────────────────
    saved = 0
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc=f"  Generating {label_str}") as pbar:
            for result in pool.imap_unordered(_worker, tasks, chunksize=4):
                if result:
                    saved += 1
                pbar.update(1)

    print(f"→ {label_str}: {saved} images saved  →  {save_dir}")
    return saved


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    dataset_root = Path(args.dataset)
    out_dir      = Path(args.out)
    max_images   = args.images_per_class
    use_wavelet  = args.wavelet or None
    num_workers  = args.workers

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_root}\n"
            "Set --dataset to your button-tone-sz directory."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  EEG Mel-Spectrogram Generator  (v4 — exact notebook copy)")
    print("=" * 60)
    print(f"  Source       : how-to-make-spectrogram-from-eeg.ipynb")
    print(f"  hop_length   : len(x) // 256   (notebook exact)")
    print(f"  width crop   : (mel_spec.shape[1]//32)*32")
    print(f"  dB scale     : librosa.power_to_db(ref=np.max)")
    print(f"  Standardise  : (mel_db + 40) / 40")
    print(f"  Wavelet      : {use_wavelet or 'disabled'}")
    print(f"  Workers      : {num_workers}")
    print(f"  Images/class : {max_images}")
    print(f"  Output       : {out_dir}")
    print("=" * 60)

    # ── Load subject → label mapping ─────────────────────────────────────────
    demo_path = dataset_root / "demographic.csv"
    if not demo_path.exists():
        raise FileNotFoundError(f"demographic.csv not found in {dataset_root}")

    demo = pd.read_csv(demo_path)
    demo.columns = demo.columns.str.strip()
    print("demographic.csv columns:", list(demo.columns))
    print(demo.head())

    diag        = dict(zip(demo["subject"], demo["group"]))  # 0=HC, 1=SZ
    hc_subjects = sorted(s for s, g in diag.items() if g == 0)
    sz_subjects = sorted(s for s, g in diag.items() if g == 1)

    print(f"\nHC subjects ({len(hc_subjects)}): {hc_subjects}")
    print(f"SZ subjects ({len(sz_subjects)}): {sz_subjects}")

    n_hc = generate_images_for_subjects(
        dataset_root, hc_subjects, "HC",
        out_dir, max_images, use_wavelet, num_workers,
    )
    n_sz = generate_images_for_subjects(
        dataset_root, sz_subjects, "SZ",
        out_dir, max_images, use_wavelet, num_workers,
    )

    print("\n" + "━" * 32)
    print(f"HC Images  : {n_hc}")
    print(f"SZ Images  : {n_sz}")
    print(f"Total      : {n_hc + n_sz}")
    print("━" * 32)
    print(f"All images saved under: {out_dir}")


if __name__ == "__main__":
    # Required by multiprocessing on spawn-based systems (Windows / Kaggle)
    main()
