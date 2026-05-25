"""
train_efficientnet.py
=====================
Trains an EfficientNetB0 classifier on the generated EEG spectrogram images.
Supports pre-extracting and caching features to save time.

Dataset layout expected
-----------------------
  eeg_images/
      HC/   *.png   (Healthy Control)
      SZ/   *.png   (Schizophrenia)

Split   : 80% train / 20% test  (stratified)
Model   : EfficientNetB0 (pretrained on ImageNet) + custom head
Input   : 128 × 256 RGB spectrogram images
Output  : Binary (0 = HC, 1 = SZ)

Usage
-----
  python train_efficientnet.py --data ./eeg_images --epochs 30
"""

# ── Suppress harmless TF/CUDA double-registration warnings ───────────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import warnings
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="/kaggle/working/eeg_images",
                   help="Path to eeg_images/ directory (contains HC/ and SZ/)")
    p.add_argument("--out",        default="/kaggle/working/model_output",
                   help="Directory to save model, plots, and results")
    p.add_argument("--features_dir", default=None,
                   help="Directory to cache/load extracted features")
    p.add_argument("--img_h",      type=int, default=128)
    p.add_argument("--img_w",      type=int, default=256)
    p.add_argument("--batch",      type=int, default=32)
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--fine_tune",  type=int, default=10,
                   help="Number of top EfficientNet layers to unfreeze for fine-tuning (0=skip)")
    args, _ = p.parse_known_args()
    return args


# ── Data loading ──────────────────────────────────────────────────────────────
def load_dataset(data_dir: Path, img_h: int, img_w: int):
    paths, labels = [], []
    class_map = {"HC": 0, "SZ": 1}

    for cls, lbl in class_map.items():
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"[WARN] {cls_dir} not found — skipping")
            continue
        imgs = sorted(cls_dir.glob("*.png"))
        paths.extend(imgs)
        labels.extend([lbl] * len(imgs))
        print(f"  {cls}: {len(imgs)} images")

    print(f"\nTotal: {len(paths)} images  |  HC={labels.count(0)}  SZ={labels.count(1)}")

    from PIL import Image
    X, y = [], []
    for p, lbl in zip(paths, labels):
        try:
            img = Image.open(p).convert("RGB").resize((img_w, img_h))
            X.append(np.array(img, dtype=np.float32))
            y.append(lbl)
        except Exception as exc:
            print(f"  [WARN] Could not load {p}: {exc}")

    X = np.stack(X)          # (N, H, W, 3)
    y = np.array(y, dtype=np.int32)
    return X, y, list(class_map.keys())


# ── Model Builders ────────────────────────────────────────────────────────────
def build_feature_extractor(img_h: int, img_w: int) -> Model:
    """EfficientNetB0 (fully frozen) -> GAP features."""
    inputs = layers.Input(shape=(img_h, img_w, 3))
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    backbone.trainable = False
    gap = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    return Model(inputs, gap, name="EfficientNetB0_FeatureExtractor")


def build_head_model(input_dim=1280) -> Model:
    """Classification head trained directly on pre-extracted features."""
    inputs = layers.Input(shape=(input_dim,), name="input_features")
    x = layers.BatchNormalization(name="bn_head")(inputs)
    x = layers.Dropout(0.4, name="drop1")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return Model(inputs, outputs, name="EfficientNetB0_EEG_Head")


def build_model(img_h: int, img_w: int, fine_tune_layers: int = 10) -> Model:
    """Full end-to-end model (for fine-tuning on images)."""
    inputs = layers.Input(shape=(img_h, img_w, 3), name="input_image")
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    backbone.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.4, name="drop1")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="EfficientNetB0_EEG")
    return model, backbone


def unfreeze_top(backbone, n: int):
    backbone.trainable = True
    for layer in backbone.layers[:-n]:
        layer.trainable = False
    frozen = sum(1 for l in backbone.layers if not l.trainable)
    total  = len(backbone.layers)
    print(f"  Fine-tuning: {n} layers unfrozen  ({total - frozen}/{total} trainable)")


# ── Training Callbacks ────────────────────────────────────────────────────────
def get_callbacks(out_dir: Path, phase: str) -> list:
    return [
        callbacks.ModelCheckpoint(
            str(out_dir / f"best_{phase}.keras"),
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1,
        ),
        callbacks.CSVLogger(str(out_dir / f"history_{phase}.csv")),
    ]


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_history(history, out_dir: Path, phase: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(axes,
                                  [("accuracy", "val_accuracy"),
                                   ("loss", "val_loss")],
                                  ["Accuracy", "Loss"]):
        ax.plot(history.history[metric[0]], label=f"Train {title}")
        ax.plot(history.history[metric[1]], label=f"Val {title}")
        ax.set_title(f"{title} — {phase}")
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    fig.savefig(str(out_dir / f"curves_{phase}.png"), dpi=100)
    plt.close(fig)
    print(f"  Saved: curves_{phase}.png")


def plot_confusion(y_true, y_pred, class_names, out_dir: Path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN + 1e-9)
    specificity = TN / (TN + FP + 1e-9)
    precision   = TP / (TP + FP + 1e-9)
    f1          = 2 * precision * sensitivity / (precision + sensitivity + 1e-9)
    n_total     = cm.sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Confusion Matrix — EfficientNetB0 (Test Set)",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, data, title, fmt, vmax in [
        (axes[0], cm,      "Raw Counts",          "d",   None),
        (axes[1], cm_norm, "Row-Normalised (%)", ".2%",  1.0 ),
    ]:
        im = ax.imshow(data, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, fontsize=11)
        ax.set_yticks(tick_marks); ax.set_yticklabels(class_names, fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label",      fontsize=11)
        ax.set_title(title, fontsize=12, pad=8)

        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = data[i, j]
                txt = f"{val:d}\n({val/n_total*100:.1f}%)" if fmt == "d" else f"{val:{fmt}}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if val > thresh else "black")

    plt.tight_layout()
    fig.savefig(str(out_dir / "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: confusion_matrix.png")

    summary_lines = [
        "",
        "╔══════════════════════════════════════╗",
        "║   Confusion Matrix  —  Test Set      ║",
        "╠══════════════════════════════════════╣",
        f"║  True Positives  (TP)  : {TP:>6d}       ║",
        f"║  True Negatives  (TN)  : {TN:>6d}       ║",
        f"║  False Positives (FP)  : {FP:>6d}       ║",
        f"║  False Negatives (FN)  : {FN:>6d}       ║",
        "╠══════════════════════════════════════╣",
        f"║  Sensitivity (SZ Rec.) : {sensitivity:>7.4f}      ║",
        f"║  Specificity (HC Rec.) : {specificity:>7.4f}      ║",
        f"║  Precision             : {precision:>7.4f}      ║",
        f"║  F1-Score              : {f1:>7.4f}      ║",
        "╚══════════════════════════════════════╝",
        "",
    ]
    for line in summary_lines:
        print(line)

    report_path = out_dir / "confusion_matrix_report.txt"
    with open(str(report_path), "w") as fh:
        fh.write("\n".join(summary_lines))
    print(f"  Saved: confusion_matrix_report.txt")


def plot_roc(y_true, y_prob, out_dir: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="royalblue", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_dir / "roc_curve.png"), dpi=100)
    plt.close()
    print("  Saved: roc_curve.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args    = parse_args()
    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set up feature caching directory
    if args.features_dir is None:
        features_dir = data_dir.parent / "features_cache"
    else:
        features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    IMG_H  = args.img_h
    IMG_W  = args.img_w
    BATCH  = args.batch
    EPOCHS = args.epochs
    LR     = args.lr

    print(f"TensorFlow version : {tf.__version__}")
    print(f"GPU available      : {tf.config.list_physical_devices('GPU')}")
    print(f"Data directory     : {data_dir}")
    print(f"Features cache dir : {features_dir}")
    print(f"Output directory   : {out_dir}\n")

    # ── 1. Check/load cached features ─────────────────────────────────────────
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
        X_train = np.load(cache_files["train_x"])
        y_train = np.load(cache_files["train_y"])
        X_val   = np.load(cache_files["val_x"])
        y_val   = np.load(cache_files["val_y"])
        X_test  = np.load(cache_files["test_x"])
        y_test  = np.load(cache_files["test_y"])
    else:
        print("[CACHE] Pre-extracted features not found. Extracting features now ...")
        print("Loading images from dataset...")
        X_img, y_img, class_names = load_dataset(data_dir, IMG_H, IMG_W)

        # 80 / 20 stratified split
        X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
            X_img, y_img, test_size=0.20, random_state=SEED, stratify=y_img
        )
        # Train / val split (80/20 of train = 64/16 overall)
        X_train_img, X_val_img, y_train_img, y_val_img = train_test_split(
            X_train_img, y_train_img, test_size=0.20, random_state=SEED, stratify=y_train_img
        )

        print("\nExtracting features using EfficientNetB0 ...")
        extractor = build_feature_extractor(IMG_H, IMG_W)
        
        # Batch extraction to prevent OOM
        def extract_all(X_data):
            feats = []
            for start in range(0, len(X_data), BATCH):
                end = min(start + BATCH, len(X_data))
                feats.append(extractor.predict(X_data[start:end], verbose=0))
            return np.vstack(feats)

        X_train = extract_all(X_train_img)
        X_val   = extract_all(X_val_img)
        X_test  = extract_all(X_test_img)
        
        y_train = y_train_img
        y_val   = y_val_img
        y_test  = y_test_img

        # Save to disk
        np.save(cache_files["train_x"], X_train)
        np.save(cache_files["train_y"], y_train)
        np.save(cache_files["val_x"],   X_val)
        np.save(cache_files["val_y"],   y_val)
        np.save(cache_files["test_x"],  X_test)
        np.save(cache_files["test_y"],  y_test)
        print(f"Features saved successfully under: {features_dir}")

        # Free image arrays
        del X_img, X_train_img, X_val_img, X_test_img
        import gc; gc.collect()

    print(f"\nFeature split summary:")
    print(f"  Train features : {X_train.shape}  (HC={np.sum(y_train==0)}, SZ={np.sum(y_train==1)})")
    print(f"  Val features   : {X_val.shape}   (HC={np.sum(y_val==0)},  SZ={np.sum(y_val==1)})")
    print(f"  Test features  : {X_test.shape}  (HC={np.sum(y_test==0)}, SZ={np.sum(y_test==1)})")

    # ── 2. Phase 1 — Feature Extraction (Train classification head directly on features) ──
    print("\n" + "=" * 50)
    print("Phase 1: Training Classification Head on Features …")
    
    head_model = build_head_model(input_dim=X_train.shape[1])
    head_model.summary(line_length=90)
    
    head_model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    hist1 = head_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=get_callbacks(out_dir, "phase1"),
        verbose=1,
    )
    plot_history(hist1, out_dir, "phase1")

    # Keep track of active model
    active_model = head_model

    # ── 3. Phase 2 — Fine-tuning (Requires full images to fine-tune backbone) ──
    if args.fine_tune > 0:
        print("\n" + "=" * 50)
        print(f"Phase 2: Fine-tuning (loading full images to unfreeze top {args.fine_tune} layers) …")
        
        # Load images for training
        X_img, y_img, _ = load_dataset(data_dir, IMG_H, IMG_W)
        X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
            X_img, y_img, test_size=0.20, random_state=SEED, stratify=y_img
        )
        X_train_img, X_val_img, y_train_img, y_val_img = train_test_split(
            X_train_img, y_train_img, test_size=0.20, random_state=SEED, stratify=y_train_img
        )

        # Build full end-to-end model
        full_model, backbone = build_model(IMG_H, IMG_W, args.fine_tune)
        
        # Copy head weights from Phase 1
        print("Transferring trained head weights from Phase 1 ...")
        for layer in head_model.layers:
            if layer.name in ["bn_head", "fc1", "output"]:
                full_model.get_layer(layer.name).set_weights(layer.get_weights())

        # Unfreeze and compile
        unfreeze_top(backbone, args.fine_tune)
        full_model.compile(
            optimizer=optimizers.Adam(learning_rate=LR / 10),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # Image generators (train only uses augmentation)
        train_gen = ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05,
            rescale=1.0,
        )
        val_gen  = ImageDataGenerator(rescale=1.0)
        
        train_flow = train_gen.flow(X_train_img, y_train_img, batch_size=BATCH, seed=SEED)
        val_flow   = val_gen.flow(X_val_img, y_val_img, batch_size=BATCH, shuffle=False)

        hist2 = full_model.fit(
            train_flow,
            validation_data=val_flow,
            epochs=EPOCHS // 2,
            callbacks=get_callbacks(out_dir, "phase2"),
            verbose=1,
        )
        plot_history(hist2, out_dir, "phase2")
        active_model = full_model

        # Free GPU memory
        del X_img, X_train_img, X_val_img, X_test_img
        import gc; gc.collect()

    # ── 4. Evaluate on TEST set ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Evaluating on test set …")
    
    if args.fine_tune > 0:
        # Full model uses test image data
        _, _, _ = load_dataset(data_dir, IMG_H, IMG_W)  # Reloading test images if needed
        # Alternatively, we can just load test images specifically. To keep it robust:
        X_img, y_img, _ = load_dataset(data_dir, IMG_H, IMG_W)
        _, X_test_img, _, y_test_img = train_test_split(
            X_img, y_img, test_size=0.20, random_state=SEED, stratify=y_img
        )
        test_gen = ImageDataGenerator(rescale=1.0)
        test_flow = test_gen.flow(X_test_img, y_test_img, batch_size=BATCH, shuffle=False)
        test_loss, test_acc = active_model.evaluate(test_flow, verbose=0)
        y_prob = active_model.predict(test_flow, verbose=0).ravel()
        y_test_final = y_test_img
    else:
        # Head model uses pre-extracted test features
        test_loss, test_acc = active_model.evaluate(X_test, y_test, verbose=0)
        y_prob = active_model.predict(X_test, verbose=0).ravel()
        y_test_final = y_test

    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")

    y_pred = (y_prob >= 0.5).astype(int)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_final, y_pred, target_names=class_names))

    # AUC
    auc = roc_auc_score(y_test_final, y_prob)
    print(f"  AUC-ROC : {auc:.4f}")

    # ── 5. Save plots & model ─────────────────────────────────────────────────
    plot_confusion(y_test_final, y_pred, class_names, out_dir)
    plot_roc(y_test_final, y_prob, out_dir)

    active_model.save(str(out_dir / "efficientnet_eeg_final.keras"))
    print(f"\n✅ Model saved → {out_dir / 'efficientnet_eeg_final.keras'}")

    # ── 6. Summary card ───────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print(f"  Train features/images : {len(X_train)}")
    print(f"  Val features/images   : {len(X_val)}")
    print(f"  Test features/images  : {len(X_test)}")
    print(f"  Test Acc     : {test_acc * 100:.2f}%")
    print(f"  AUC-ROC      : {auc:.4f}")
    print("━" * 50)
    print(f"All outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
