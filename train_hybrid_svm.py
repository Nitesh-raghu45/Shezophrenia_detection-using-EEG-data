"""
train_hybrid_svm.py
====================
Hybrid EEG Schizophrenia Classifier
  Stage 1 : EfficientNetB0 (ImageNet, frozen) → 1280-dim feature vectors
  Stage 2 : Support Vector Machine (RBF / Linear / Poly kernel)

Usage
-----
  python train_hybrid_svm.py --data ./eeg_images --clf_kernel rbf --C 5.0
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import sys
import time
import argparse
import warnings
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    ConfusionMatrixDisplay, accuracy_score,
)

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model, layers

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid EfficientNetB0 + SVM EEG Classifier"
    )
    p.add_argument("--data",    default="/kaggle/working/eeg_images",
                   help="Path to eeg_images/ (HC/ and SZ/ sub-folders)")
    p.add_argument("--out",     default="/kaggle/working/svm_output",
                   help="Directory to save model, plots, and results")
    p.add_argument("--features_dir", default=None,
                   help="Directory to cache/load extracted features")
    p.add_argument("--img_h",   type=int, default=128)
    p.add_argument("--img_w",   type=int, default=256)
    p.add_argument("--batch",   type=int, default=64,
                   help="Batch size for feature extraction")

    # SVM hyper-parameters
    p.add_argument("--clf_kernel", default="rbf",
                   choices=["rbf", "linear", "poly", "sigmoid"],
                   help="SVM kernel (default: rbf)")
    p.add_argument("--C",       type=float, default=1.0,
                   help="SVM regularisation  (default: 1.0)")
    p.add_argument("--gamma",   default="scale",
                   help="SVM gamma for rbf/poly/sigmoid (default: scale)")
    p.add_argument("--degree",  type=int, default=3,
                   help="Polynomial degree when kernel=poly (default: 3)")

    # PCA
    p.add_argument("--pca",     type=int, default=256,
                   help="PCA components (0 = skip PCA, default: 256)")

    # Optional extras
    p.add_argument("--cv",          type=int, default=5,
                   help="Stratified k-fold CV folds (0 = skip)")
    p.add_argument("--grid_search", action="store_true",
                   help="Run GridSearchCV to find best C and gamma")
    p.add_argument("--compare_kernels", action="store_true",
                   help="Train RBF, Linear, and Poly SVMs and compare")

    args, _ = p.parse_known_args()
    return args


# ── Data loading ──────────────────────────────────────────────────────────────
def load_dataset(data_dir: Path, img_h: int, img_w: int):
    paths, labels = [], []
    class_map = {"HC": 0, "SZ": 1}

    for cls, lbl in class_map.items():
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"  [WARN] {cls_dir} not found — skipping")
            continue
        imgs = sorted(cls_dir.glob("*.png"))
        paths.extend(imgs)
        labels.extend([lbl] * len(imgs))
        print(f"  {cls}: {len(imgs)} images")

    if len(paths) == 0:
        print(f"\n[ERROR] No images found in {data_dir}")
        print("  Make sure HC/ and SZ/ sub-folders exist with PNG files.")
        sys.exit(1)

    print(f"\n  Total: {len(paths)} images  | HC={labels.count(0)}  SZ={labels.count(1)}")

    X, y = [], []
    for p, lbl in tqdm(zip(paths, labels), total=len(paths), desc="  Loading images"):
        try:
            img = Image.open(p).convert("RGB").resize((img_w, img_h))
            X.append(np.array(img, dtype=np.float32))
            y.append(lbl)
        except Exception as exc:
            print(f"  [WARN] Could not load {p}: {exc}")

    if len(X) == 0:
        print("[ERROR] No images could be loaded.")
        sys.exit(1)

    return np.stack(X), np.array(y, dtype=np.int32), list(class_map.keys())


# ── Feature extractor ─────────────────────────────────────────────────────────
def build_feature_extractor(img_h: int, img_w: int) -> Model:
    inputs   = layers.Input(shape=(img_h, img_w, 3))
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    backbone.trainable = False
    gap = layers.GlobalAveragePooling2D()(backbone.output)
    return Model(inputs, gap, name="EfficientNetB0_FeatureExtractor")


def extract_features(model: Model, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    feats = []
    for start in tqdm(range(0, len(X), batch_size), desc="  Extracting features"):
        batch = X[start: start + batch_size]
        feats.append(model.predict(batch, verbose=0))
    return np.vstack(feats).astype(np.float32)


# ── SVM pipeline builder ──────────────────────────────────────────────────────
def build_svm_pipeline(kernel: str, C: float, gamma, degree: int, pca_components: int) -> Pipeline:
    steps = [("scaler", StandardScaler())]

    if pca_components > 0:
        steps.append(("pca", PCA(n_components=pca_components,
                                 random_state=SEED,
                                 svd_solver="randomized")))

    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        probability=True,
        class_weight="balanced",
        random_state=SEED,
        cache_size=2000,
    )
    steps.append(("svm", svm))
    return Pipeline(steps)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred, class_names, out_dir, tag=""):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — SVM ({tag})")
    plt.tight_layout()
    fname = f"confusion_matrix_svm_{tag}.png"
    fig.savefig(str(out_dir / fname), dpi=100)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_roc(y_true, y_prob, out_dir, tag=""):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="royalblue", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — SVM ({tag})")
    ax.legend()
    plt.tight_layout()
    fname = f"roc_curve_svm_{tag}.png"
    fig.savefig(str(out_dir / fname), dpi=100)
    plt.close(fig)
    print(f"  Saved: {fname}")
    return auc


def plot_pca_variance(pipeline, out_dir):
    if "pca" not in pipeline.named_steps:
        return
    pca     = pipeline.named_steps["pca"]
    cumvar  = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, marker=".", color="darkorange")
    ax.axhline(0.95, ls="--", c="red",  lw=1, label="95% variance")
    ax.axhline(0.99, ls="--", c="blue", lw=1, label="99% variance")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Explained Variance")
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(out_dir / "pca_variance_svm.png"), dpi=100)
    plt.close(fig)
    print("  Saved: pca_variance_svm.png")


def plot_kernel_comparison(results: dict, out_dir: Path):
    kernels  = list(results.keys())
    accs     = [results[k]["accuracy"] for k in kernels]
    aucs     = [results[k]["auc"]      for k in kernels]

    x   = np.arange(len(kernels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="steelblue")
    bars2 = ax.bar(x + w/2, aucs, w, label="AUC-ROC",  color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in kernels])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("SVM Kernel Comparison")
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    fig.savefig(str(out_dir / "kernel_comparison.png"), dpi=100)
    plt.close(fig)
    print("  Saved: kernel_comparison.png")


# ── Grid search ───────────────────────────────────────────────────────────────
def run_grid_search(F_train, y_train, pca_components, kernel):
    print(f"\n[GridSearchCV] Tuning C and gamma for kernel={kernel} …")
    param_grid = {
        "svm__C":     [0.1, 1.0, 5.0, 10.0, 50.0],
        "svm__gamma": ["scale", "auto", 0.001, 0.01],
    }
    if kernel == "linear":
        param_grid = {"svm__C": [0.1, 1.0, 5.0, 10.0]}

    pipe = build_svm_pipeline(
        kernel=kernel, C=1.0, gamma="scale", degree=3,
        pca_components=pca_components,
    )
    cv   = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    gs   = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1)
    gs.fit(F_train, y_train)

    print(f"  Best params  : {gs.best_params_}")
    print(f"  Best AUC(CV) : {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_


# ── Single SVM run ────────────────────────────────────────────────────────────
def run_svm(F_train, y_train, F_test, y_test, class_names, out_dir, args, tag=None) -> dict:
    kernel = tag or args.clf_kernel
    print(f"\n{'='*55}")
    print(f"  SVM  kernel={kernel}  C={args.C}  gamma={args.gamma}  pca={args.pca if args.pca > 0 else 'off'}")
    print(f"{'='*55}")

    if args.grid_search and not tag:
        pipeline, best_params = run_grid_search(F_train, y_train, args.pca, kernel)
        C_used     = best_params.get("svm__C",     args.C)
        gamma_used = best_params.get("svm__gamma", args.gamma)
    else:
        C_used     = args.C
        gamma_used = args.gamma
        pipeline   = build_svm_pipeline(
            kernel=kernel, C=C_used, gamma=gamma_used,
            degree=args.degree, pca_components=args.pca,
        )

    if args.cv > 0:
        print(f"\n[CV] {args.cv}-fold Stratified CV on train set …")
        cv       = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED)
        t0       = time.time()
        cv_accs  = cross_val_score(pipeline, F_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_aucs  = cross_val_score(pipeline, F_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"  CV Accuracy : {cv_accs.mean():.4f} ± {cv_accs.std():.4f}")
        print(f"  CV AUC-ROC  : {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
        print(f"  CV Time     : {time.time()-t0:.1f}s")

    print("\n[Train] Fitting SVM …")
    t0 = time.time()
    pipeline.fit(F_train, y_train)
    print(f"  Training time : {time.time()-t0:.1f}s")

    svm_clf = pipeline.named_steps["svm"]
    if hasattr(svm_clf, "n_support_"):
        print(f"  Support vectors : HC={svm_clf.n_support_[0]}  SZ={svm_clf.n_support_[1]}  total={sum(svm_clf.n_support_)}")

    print("\n[Eval] Testing …")
    y_pred = pipeline.predict(F_test)
    y_prob = pipeline.predict_proba(F_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n  Test Accuracy : {acc * 100:.2f}%")
    print(f"  AUC-ROC       : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    plot_confusion(y_test, y_pred, class_names, out_dir, tag=kernel)
    plot_roc(y_test, y_prob, out_dir, tag=kernel)
    plot_pca_variance(pipeline, out_dir)

    model_path = out_dir / f"svm_{kernel}_pipeline.pkl"
    joblib.dump(pipeline, str(model_path))
    print(f"\n  Pipeline saved → {model_path}")

    return {"accuracy": acc, "auc": auc}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args     = parse_args()
    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set up feature caching directory
    if args.features_dir is None:
        features_dir = data_dir.parent / "features_cache"
    else:
        features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Hybrid EfficientNetB0 + SVM  —  EEG Classifier")
    print(f"  Kernel       : {args.clf_kernel.upper()}")
    print(f"  C            : {args.C}")
    print(f"  Gamma        : {args.gamma}")
    print(f"  PCA dims     : {args.pca if args.pca > 0 else 'disabled'}")
    print(f"  Features     : {features_dir}")
    print(f"  TF version   : {tf.__version__}")
    print(f"  GPU          : {tf.config.list_physical_devices('GPU')}")
    print("=" * 55)

    # Check if cached features exist (share with EfficientNet cache)
    cache_files = {
        "train_x": features_dir / "train_features_1280.npy",
        "train_y": features_dir / "train_labels.npy",
        "val_x":   features_dir / "val_features_1280.npy",
        "val_y":   features_dir / "val_labels.npy",
        "test_x":  features_dir / "test_features_1280.npy",
        "test_y":  features_dir / "test_labels.npy",
    }
    
    cache_exists = all(f.exists() for f in cache_files.values())
    class_names = ["HC", "SZ"]

    if cache_exists:
        print("[CACHE] Loading pre-extracted features from disk ...")
        F_train_part = np.load(cache_files["train_x"])
        y_train_part = np.load(cache_files["train_y"])
        F_val_part   = np.load(cache_files["val_x"])
        y_val_part   = np.load(cache_files["val_y"])
        F_test       = np.load(cache_files["test_x"])
        y_test       = np.load(cache_files["test_y"])

        # Combine train & val to get back the full 80% training set
        F_train = np.concatenate([F_train_part, F_val_part], axis=0)
        y_train = np.concatenate([y_train_part, y_val_part], axis=0)
    else:
        print("[CACHE] Pre-extracted features not found. Extracting now ...")
        X, y, class_names = load_dataset(data_dir, args.img_h, args.img_w)

        # 80/20 stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=SEED, stratify=y,
        )

        print("\nBuilding EfficientNetB0 feature extractor …")
        extractor = build_feature_extractor(args.img_h, args.img_w)

        print("\nExtracting features from TRAIN images …")
        t0 = time.time()
        F_train = extract_features(extractor, X_train, batch_size=args.batch)
        print(f"  Train features : {F_train.shape}  ({time.time()-t0:.1f}s)")

        print("\nExtracting features from TEST images …")
        t0 = time.time()
        F_test = extract_features(extractor, X_test, batch_size=args.batch)
        print(f"  Test features  : {F_test.shape}  ({time.time()-t0:.1f}s)")

        # Save to cache so other scripts can use them
        F_train_part, F_val_part, y_train_part, y_val_part = train_test_split(
            F_train, y_train, test_size=0.20, random_state=SEED, stratify=y_train
        )
        np.save(cache_files["train_x"], F_train_part)
        np.save(cache_files["train_y"], y_train_part)
        np.save(cache_files["val_x"],   F_val_part)
        np.save(cache_files["val_y"],   y_val_part)
        np.save(cache_files["test_x"],  F_test)
        np.save(cache_files["test_y"],  y_test)
        print(f"Features saved successfully to: {features_dir}")

        del X, X_train, X_test
        import gc; gc.collect()

    print(f"\nSplit summary:")
    print(f"  Train features : {F_train.shape}  (HC={np.sum(y_train==0)}, SZ={np.sum(y_train==1)})")
    print(f"  Test features  : {F_test.shape}   (HC={np.sum(y_test==0)}, SZ={np.sum(y_test==1)})")

    if args.compare_kernels:
        kernels = ["rbf", "linear", "poly"]
        results = {}
        for k in kernels:
            args_copy = argparse.Namespace(**vars(args))
            args_copy.clf_kernel = k
            args_copy.cv = 0
            results[k] = run_svm(F_train, y_train, F_test, y_test, class_names, out_dir, args_copy, tag=k)

        print("\n" + "=" * 55)
        print("  Kernel Comparison Summary")
        print("=" * 55)
        print(f"  {'Kernel':<10} {'Accuracy':>10} {'AUC-ROC':>10}")
        print(f"  {'-'*32}")
        for k, v in results.items():
            print(f"  {k.upper():<10} {v['accuracy']*100:>9.2f}%  {v['auc']:>10.4f}")

        plot_kernel_comparison(results, out_dir)
    else:
        results = run_svm(F_train, y_train, F_test, y_test, class_names, out_dir, args)

        print("\n" + "━" * 55)
        print(f"  Architecture  : EfficientNetB0 (frozen) + SVM")
        print(f"  Kernel        : {args.clf_kernel.upper()}")
        print(f"  C             : {args.C}")
        print(f"  Feature dim   : {F_train.shape[1]}  →  PCA {args.pca}")
        print(f"  Train samples : {len(F_train)}")
        print(f"  Test  samples : {len(F_test)}")
        print(f"  Test Accuracy : {results['accuracy'] * 100:.2f}%")
        print(f"  AUC-ROC       : {results['auc']:.4f}")
        print(f"  Output dir    : {out_dir}")
        print("━" * 55)


if __name__ == "__main__":
    main()
