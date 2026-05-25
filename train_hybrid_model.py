"""
train_hybrid_model.py
======================
Hybrid EEG Schizophrenia Classifier
-------------------------------------
Architecture:
  1. EfficientNetB0 (pretrained, frozen) → extracts 1280-dim deep feature vectors
  2. Classifier (choose one):
       a. Random Forest   --clf rf   (default)
       b. Support Vector Machine     --clf svm
       c. Soft Voting Ensemble (RF + SVM + LR)  --clf ensemble

Usage
-----
  python train_hybrid_model.py --data ./eeg_images --clf rf
"""

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import argparse
import warnings
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay, accuracy_score
)

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model, layers

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid EfficientNetB0 + ML EEG Classifier"
    )
    p.add_argument("--data",   default="/kaggle/working/eeg_images",
                   help="Path to eeg_images/ (contains HC/ and SZ/)")
    p.add_argument("--out",    default="/kaggle/working/hybrid_output",
                   help="Directory to save model, plots, and results")
    p.add_argument("--features_dir", default=None,
                   help="Directory to cache/load extracted features")
    p.add_argument("--img_h",  type=int, default=128)
    p.add_argument("--img_w",  type=int, default=256)
    p.add_argument("--batch",  type=int, default=64,
                   help="Batch size for feature extraction")
    p.add_argument("--clf",    default="rf",
                   choices=["rf", "svm", "ensemble"],
                   help="Classifier: rf | svm | ensemble (rf+svm+lr voting)")
    p.add_argument("--pca",    type=int, default=256,
                   help="PCA components to keep (0 = skip PCA)")
    p.add_argument("--n_est",  type=int, default=500,
                   help="[RF] n_estimators (default 500)")
    p.add_argument("--max_depth", type=int, default=None,
                   help="[RF] max_depth (default None = unlimited)")
    p.add_argument("--svm_c",  type=float, default=1.0,
                   help="[SVM] Regularisation C (default 1.0)")
    p.add_argument("--svm_kernel", default="rbf",
                   choices=["rbf", "linear", "poly"],
                   help="[SVM] Kernel (default rbf)")
    p.add_argument("--cv",     type=int, default=5,
                   help="Stratified k-fold CV folds (0 = skip CV)")
    args, _ = p.parse_known_args()
    return args


# ── Data loading ───────────────────────────────────────────────────────────────
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

    print(f"\nTotal: {len(paths)} images  |  HC={labels.count(0)}  SZ={labels.count(1)}")

    X, y = [], []
    for p, lbl in tqdm(zip(paths, labels), total=len(paths), desc="  Loading images"):
        try:
            img = Image.open(p).convert("RGB").resize((img_w, img_h))
            X.append(np.array(img, dtype=np.float32))
            y.append(lbl)
        except Exception as exc:
            print(f"  [WARN] Could not load {p}: {exc}")

    if len(X) == 0:
        print("\n[ERROR] No images were loaded. Check that HC/ and SZ/ contain .png files.")
        sys.exit(1)

    return np.stack(X), np.array(y, dtype=np.int32), list(class_map.keys())


# ── Feature extractor (EfficientNetB0 GAP layer) ──────────────────────────────
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
    n = len(X)
    for start in tqdm(range(0, n, batch_size), desc="  Extracting features"):
        batch = X[start : start + batch_size]
        feats.append(model.predict(batch, verbose=0))
    return np.vstack(feats).astype(np.float32)


# ── Classifier builders ────────────────────────────────────────────────────────
def build_random_forest(n_estimators: int = 500, max_depth=None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        oob_score=True,
    )


def build_svm(C: float = 1.0, kernel: str = "rbf") -> SVC:
    return SVC(
        C=C,
        kernel=kernel,
        probability=True,
        class_weight="balanced",
        random_state=SEED,
    )


def build_ensemble() -> VotingClassifier:
    rf  = build_random_forest(n_estimators=300)
    svm = build_svm(C=1.0, kernel="rbf")
    lr  = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=SEED, C=0.5)
    return VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("lr", lr)],
        voting="soft",
        weights=[3, 2, 1],
        n_jobs=-1,
    )


# ── Pipeline factory ───────────────────────────────────────────────────────────
def build_pipeline(clf_name: str, pca_components: int, args) -> Pipeline:
    steps = [("scaler", StandardScaler())]

    if pca_components > 0:
        steps.append(("pca", PCA(n_components=pca_components,
                                 random_state=SEED,
                                 svd_solver="randomized")))
        print(f"  PCA: keeping {pca_components} components")

    if clf_name == "rf":
        clf = build_random_forest(args.n_est, args.max_depth)
        print(f"  Classifier: Random Forest  (n_estimators={args.n_est}, max_depth={args.max_depth})")
    elif clf_name == "svm":
        clf = build_svm(args.svm_c, args.svm_kernel)
        print(f"  Classifier: SVM  (C={args.svm_c}, kernel={args.svm_kernel})")
    else:
        clf = build_ensemble()
        print("  Classifier: Soft-Voting Ensemble  (RF + SVM + LR)")

    steps.append(("clf", clf))
    return Pipeline(steps)


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred, class_names, out_dir: Path, tag: str = ""):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {tag}")
    plt.tight_layout()
    fname = f"confusion_matrix_{tag}.png" if tag else "confusion_matrix.png"
    fig.savefig(str(out_dir / fname), dpi=100)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_roc(y_true, y_prob, out_dir: Path, tag: str = ""):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="royalblue", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {tag}")
    plt.legend()
    plt.tight_layout()
    fname = f"roc_curve_{tag}.png" if tag else "roc_curve.png"
    plt.savefig(str(out_dir / fname), dpi=100)
    plt.close()
    print(f"  Saved: {fname}")


def plot_rf_feature_importance(pipeline: Pipeline, out_dir: Path,
                                pca_components: int, feat_dim: int):
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    top_imp = importances[indices]

    if pca_components > 0:
        labels = [f"PC{i+1}" for i in indices]
        title  = "Top-20 PCA Component Importances (RF)"
    else:
        labels = [f"F{i}" for i in indices]
        title  = "Top-20 Feature Importances (RF)"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(indices)), top_imp, color="steelblue")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Importance")
    plt.tight_layout()
    fig.savefig(str(out_dir / "rf_feature_importance.png"), dpi=100)
    plt.close(fig)
    print("  Saved: rf_feature_importance.png")


def plot_pca_variance(pipeline: Pipeline, out_dir: Path):
    if "pca" not in pipeline.named_steps:
        return
    pca = pipeline.named_steps["pca"]
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, marker=".", color="darkorange")
    ax.axhline(0.95, ls="--", c="red",  lw=1, label="95% variance")
    ax.axhline(0.99, ls="--", c="blue", lw=1, label="99% variance")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Explained Variance")
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(out_dir / "pca_variance.png"), dpi=100)
    plt.close(fig)
    print("  Saved: pca_variance.png")


# ── Main ───────────────────────────────────────────────────────────────────────
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

    print("=" * 60)
    print("  Hybrid EfficientNetB0 + ML Classifier")
    print(f"  Classifier   : {args.clf.upper()}")
    print(f"  PCA dims     : {args.pca if args.pca > 0 else 'disabled'}")
    print(f"  Data         : {data_dir}")
    print(f"  Features     : {features_dir}")
    print(f"  Output       : {out_dir}")
    print(f"  TF version   : {tf.__version__}")
    print(f"  GPU          : {tf.config.list_physical_devices('GPU')}")
    print("=" * 60)

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
        # Load images and extract
        X, y, class_names = load_dataset(data_dir, args.img_h, args.img_w)

        # 80/20 stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=SEED, stratify=y
        )

        print("\nBuilding EfficientNetB0 feature extractor …")
        extractor = build_feature_extractor(args.img_h, args.img_w)
        
        print("\nExtracting deep features from TRAIN images …")
        t0 = time.time()
        F_train = extract_features(extractor, X_train, batch_size=args.batch)
        print(f"  Train features shape: {F_train.shape}  ({time.time()-t0:.1f}s)")

        print("\nExtracting deep features from TEST images …")
        t0 = time.time()
        F_test = extract_features(extractor, X_test, batch_size=args.batch)
        print(f"  Test  features shape: {F_test.shape}  ({time.time()-t0:.1f}s)")

        # Save to cache so other scripts can use them
        # (Save train directly as train_features_1280.npy and save val as empty/dummy if needed, or split 80% train to create matching val cache)
        # To match train_efficientnet split: split 80% train into 64% train and 16% val
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

    # ── 2. Build ML pipeline ───────────────────────────────────────────────────
    print("\n[Step 4] Building ML pipeline …")
    pipeline = build_pipeline(args.clf, args.pca, args)

    # ── 3. (Optional) Cross-validation ────────────────────────────────────────
    if args.cv > 0:
        print(f"\n[Step 5] {args.cv}-fold Stratified Cross-Validation on TRAIN set …")
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED)
        t0 = time.time()
        cv_scores = cross_val_score(
            pipeline, F_train, y_train,
            cv=cv, scoring="accuracy", n_jobs=-1,
        )
        print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  CV Scores   : {np.round(cv_scores, 4)}")
        print(f"  CV Time     : {time.time()-t0:.1f}s")

    # ── 4. Train on full train set ─────────────────────────────────────────────
    print("\n[Step 6] Training on full train set …")
    t0 = time.time()
    pipeline.fit(F_train, y_train)
    print(f"  Training time: {time.time()-t0:.1f}s")

    # OOB score (Random Forest only)
    if args.clf == "rf":
        rf_clf = pipeline.named_steps["clf"]
        if hasattr(rf_clf, "oob_score_"):
            print(f"  OOB Accuracy (RF): {rf_clf.oob_score_:.4f}")

    # ── 5. Evaluate on test set ────────────────────────────────────────────────
    print("\n[Step 7] Evaluating on TEST set …")
    y_pred = pipeline.predict(F_test)
    y_prob = pipeline.predict_proba(F_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    auc      = roc_auc_score(y_test, y_prob)

    print(f"\n  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  AUC-ROC       : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ── 6. Save plots ──────────────────────────────────────────────────────────
    tag = args.clf
    plot_confusion(y_test, y_pred, class_names, out_dir, tag=tag)
    plot_roc(y_test, y_prob, out_dir, tag=tag)
    plot_pca_variance(pipeline, out_dir)
    if args.clf == "rf":
        plot_rf_feature_importance(pipeline, out_dir, args.pca, F_train.shape[1])

    # ── 7. Save the sklearn pipeline ─────────────────────────────────────────
    import joblib
    model_path = out_dir / f"hybrid_{args.clf}_pipeline.pkl"
    joblib.dump(pipeline, str(model_path))
    print(f"\n✅ Pipeline saved → {model_path}")

    # ── 8. Summary card ───────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print(f"  Architecture     : EfficientNetB0 (frozen) + {args.clf.upper()}")
    print(f"  Feature dim      : {F_train.shape[1]}  →  PCA {args.pca}")
    print(f"  Train samples    : {len(F_train)}")
    print(f"  Test  samples    : {len(F_test)}")
    print(f"  Test Accuracy    : {test_acc * 100:.2f}%")
    print(f"  AUC-ROC          : {auc:.4f}")
    print(f"  Output directory : {out_dir}")
    print("━" * 60)


if __name__ == "__main__":
    main()
