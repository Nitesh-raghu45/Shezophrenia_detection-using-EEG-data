"""
train_cnn_bilstm.py
===================
Hybrid CNN + Bidirectional LSTM classifier for EEG spectrogram images.
Supports pre-extracting and caching sequence features to save time.

Architecture
------------
  Input (128×256×3)
    └─ EfficientNetB0 backbone  →  feature map (4 × 8 × 1280)
    └─ Mean-pool over freq axis →  time sequence (8 × 1280)
    └─ BiLSTM(256, return_sequences=True)
    └─ BiLSTM(128)
    └─ BN → Dropout(0.4) → Dense(256, relu) → Dropout(0.3) → Dense(1, sigmoid)

Usage
-----
  python train_cnn_bilstm.py --data ./eeg_images --epochs 40
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
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

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",      default="/kaggle/working/eeg_images")
    p.add_argument("--out",       default="/kaggle/working/cnn_bilstm_output")
    p.add_argument("--features_dir", default=None,
                   help="Directory to cache/load extracted features")
    p.add_argument("--img_h",     type=int,   default=128)
    p.add_argument("--img_w",     type=int,   default=256)
    p.add_argument("--batch",     type=int,   default=16)
    p.add_argument("--epochs",    type=int,   default=40)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--fine_tune", type=int,   default=20,
                   help="Top N backbone layers to unfreeze in phase 2 (0=skip)")
    args, _ = p.parse_known_args()
    return args


# ── Data loading ──────────────────────────────────────────────────────────────
def load_dataset(data_dir: Path, img_h: int, img_w: int):
    paths, labels = [], []
    class_map = {"HC": 0, "SZ": 1}
    for cls, lbl in class_map.items():
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"[WARN] {cls_dir} not found — skipping"); continue
        imgs = sorted(cls_dir.glob("*.png"))
        paths.extend(imgs); labels.extend([lbl] * len(imgs))
        print(f"  {cls}: {len(imgs)} images")
    print(f"\nTotal: {len(paths)}  |  HC={labels.count(0)}  SZ={labels.count(1)}")

    X, y = [], []
    for p, lbl in zip(paths, labels):
        try:
            img = Image.open(p).convert("RGB").resize((img_w, img_h))
            X.append(np.array(img, dtype=np.float32))
            y.append(lbl)
        except Exception as e:
            print(f"  [WARN] {p}: {e}")
    return np.stack(X), np.array(y, dtype=np.int32), list(class_map.keys())


# ── SpecAugment (time + freq masking) ────────────────────────────────────────
class SpecAugment(layers.Layer):
    def __init__(self, time_mask=30, freq_mask=20, **kwargs):
        super().__init__(**kwargs)
        self.time_mask = time_mask
        self.freq_mask = freq_mask

    def call(self, x, training=None):
        if not training:
            return x
        shape = tf.shape(x)
        H, W = shape[1], shape[2]

        f  = tf.random.uniform((), 0, self.freq_mask, dtype=tf.int32)
        f0 = tf.random.uniform((), 0, H - f,          dtype=tf.int32)
        freq_mask = tf.concat([
            tf.ones([f0, W], dtype=x.dtype),
            tf.zeros([f,  W], dtype=x.dtype),
            tf.ones([H - f0 - f, W], dtype=x.dtype),
        ], axis=0)

        t  = tf.random.uniform((), 0, self.time_mask, dtype=tf.int32)
        t0 = tf.random.uniform((), 0, W - t,          dtype=tf.int32)
        time_mask = tf.concat([
            tf.ones([H, t0], dtype=x.dtype),
            tf.zeros([H, t],  dtype=x.dtype),
            tf.ones([H, W - t0 - t], dtype=x.dtype),
        ], axis=1)

        mask = freq_mask * time_mask
        mask = tf.expand_dims(mask, 0)
        mask = tf.expand_dims(mask, -1)
        return x * mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"time_mask": self.time_mask, "freq_mask": self.freq_mask})
        return cfg


# ── Model Builders ────────────────────────────────────────────────────────────
def build_feature_extractor_seq(img_h: int, img_w: int) -> Model:
    """EfficientNetB0 (frozen) + frequency pooling → shape (8, 1280)."""
    inputs = layers.Input(shape=(img_h, img_w, 3))
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    backbone.trainable = False
    feat = backbone.output
    # Pool over frequency axis (axis=1) → (batch, 8, 1280)
    time_seq = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1),
        name="freq_pool"
    )(feat)
    return Model(inputs, time_seq, name="EfficientNetB0_SeqExtractor")


def build_bilstm_head(input_shape=(8, 1280)) -> Model:
    """BiLSTM head model trained directly on sequence features."""
    inputs = layers.Input(shape=input_shape, name="input_seq")
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_1"
    )(inputs)
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False, dropout=0.2),
        name="bilstm_2"
    )(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.4, name="drop1")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return Model(inputs, outputs, name="BiLSTM_Head")


def build_cnn_bilstm(img_h: int, img_w: int) -> tuple[Model, object]:
    """Full end-to-end model (for fine-tuning on images)."""
    inputs = layers.Input(shape=(img_h, img_w, 3), name="input_image")
    x_aug = SpecAugment(time_mask=30, freq_mask=20, name="spec_augment")(inputs)

    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x_aug,
    )
    backbone.trainable = False
    feat = backbone.output

    time_seq = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1),
        name="freq_pool"
    )(feat)

    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_1"
    )(time_seq)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False, dropout=0.2),
        name="bilstm_2"
    )(x)

    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.4, name="drop1")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="CNN_BiLSTM_EEG")
    return model, backbone


def unfreeze_top(backbone, n: int):
    backbone.trainable = True
    for layer in backbone.layers[:-n]:
        layer.trainable = False
    trainable = sum(1 for l in backbone.layers if l.trainable)
    print(f"  Fine-tuning: top {n} layers unfrozen ({trainable}/{len(backbone.layers)} trainable)")


# ── Training Callbacks ────────────────────────────────────────────────────────
def get_callbacks(out_dir: Path, phase: str) -> list:
    return [
        callbacks.ModelCheckpoint(
            str(out_dir / f"best_{phase}.keras"),
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-7, verbose=1,
        ),
        callbacks.CSVLogger(str(out_dir / f"history_{phase}.csv")),
    ]


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_history(history, out_dir: Path, phase: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (m, vm), title in zip(
        axes,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Loss"],
    ):
        ax.plot(history.history[m],  label=f"Train {title}")
        ax.plot(history.history[vm], label=f"Val {title}")
        ax.set_title(f"{title} — {phase}")
        ax.set_xlabel("Epoch"); ax.legend()
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
    f1          = 2*precision*sensitivity / (precision + sensitivity + 1e-9)
    n_total     = cm.sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Confusion Matrix — CNN+BiLSTM (Test Set)", fontsize=14, fontweight="bold", y=1.01)

    for ax, data, title, fmt, vmax in [
        (axes[0], cm,      "Raw Counts",        "d",    None),
        (axes[1], cm_norm, "Row-Normalised (%)", ".2%", 1.0),
    ]:
        im = ax.imshow(data, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = np.arange(len(class_names))
        ax.set_xticks(ticks); ax.set_xticklabels(class_names, fontsize=11)
        ax.set_yticks(ticks); ax.set_yticklabels(class_names, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
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

    lines = [
        "", "╔══════════════════════════════════════╗",
        "║   Confusion Matrix  —  CNN+BiLSTM    ║",
        "╠══════════════════════════════════════╣",
        f"║  True Positives  (TP) : {TP:>6d}        ║",
        f"║  True Negatives  (TN) : {TN:>6d}        ║",
        f"║  False Positives (FP) : {FP:>6d}        ║",
        f"║  False Negatives (FN) : {FN:>6d}        ║",
        "╠══════════════════════════════════════╣",
        f"║  Sensitivity (SZ Rec.): {sensitivity:>7.4f}       ║",
        f"║  Specificity (HC Rec.): {specificity:>7.4f}       ║",
        f"║  Precision            : {precision:>7.4f}       ║",
        f"║  F1-Score             : {f1:>7.4f}       ║",
        "╚══════════════════════════════════════╝", "",
    ]
    for l in lines: print(l)
    with open(out_dir / "confusion_matrix_report.txt", "w") as fh:
        fh.write("\n".join(lines))
    print("  Saved: confusion_matrix_report.txt")


def plot_roc(y_true, y_prob, out_dir: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color="royalblue", lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — CNN+BiLSTM (Test Set)"); plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_dir / "roc_curve.png"), dpi=100)
    plt.close()
    print("  Saved: roc_curve.png")


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

    IMG_H  = args.img_h
    IMG_W  = args.img_w
    BATCH  = args.batch
    EPOCHS = args.epochs
    LR     = args.lr

    print(f"TensorFlow : {tf.__version__}")
    print(f"GPU        : {tf.config.list_physical_devices('GPU')}")
    print(f"Data       : {data_dir}")
    print(f"Features   : {features_dir}")
    print(f"Output     : {out_dir}\n")

    # ── 1. Check/load cached sequence features ───────────────────────────────
    cache_files = {
        "train_x": features_dir / "train_features_seq.npy",
        "train_y": features_dir / "train_labels.npy",  # share labels with EfficientNet
        "val_x":   features_dir / "val_features_seq.npy",
        "val_y":   features_dir / "val_labels.npy",
        "test_x":  features_dir / "test_features_seq.npy",
        "test_y":  features_dir / "test_labels.npy",
    }
    
    cache_exists = all(f.exists() for f in cache_files.values())
    class_names = ["HC", "SZ"]

    if cache_exists:
        print("[CACHE] Loading pre-extracted sequence features from disk ...")
        X_train = np.load(cache_files["train_x"])
        y_train = np.load(cache_files["train_y"])
        X_val   = np.load(cache_files["val_x"])
        y_val   = np.load(cache_files["val_y"])
        X_test  = np.load(cache_files["test_x"])
        y_test  = np.load(cache_files["test_y"])
    else:
        print("[CACHE] Pre-extracted sequence features not found. Extracting now ...")
        print("Loading images from dataset...")
        X_img, y_img, class_names = load_dataset(data_dir, IMG_H, IMG_W)

        # Stratified 64 / 16 / 20 split
        X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
            X_img, y_img, test_size=0.20, random_state=SEED, stratify=y_img
        )
        X_train_img, X_val_img, y_train_img, y_val_img = train_test_split(
            X_train_img, y_train_img, test_size=0.20, random_state=SEED, stratify=y_train_img
        )

        print("\nExtracting sequence features using EfficientNetB0 + freq pool ...")
        extractor = build_feature_extractor_seq(IMG_H, IMG_W)
        
        # Batch extraction to prevent OOM
        def extract_seq_all(X_data):
            feats = []
            for start in range(0, len(X_data), BATCH):
                end = min(start + BATCH, len(X_data))
                feats.append(extractor.predict(X_data[start:end], verbose=0))
            return np.vstack(feats)

        X_train = extract_seq_all(X_train_img)
        X_val   = extract_seq_all(X_val_img)
        X_test  = extract_seq_all(X_test_img)
        
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
        print(f"Sequence features saved successfully under: {features_dir}")

        # Free image arrays
        del X_img, X_train_img, X_val_img, X_test_img
        import gc; gc.collect()

    print(f"\nSequence split summary:")
    print(f"  Train sequences : {X_train.shape}  HC={np.sum(y_train==0)}  SZ={np.sum(y_train==1)}")
    print(f"  Val sequences   : {X_val.shape}   HC={np.sum(y_val==0)}   SZ={np.sum(y_val==1)}")
    print(f"  Test sequences  : {X_test.shape}  HC={np.sum(y_test==0)}  SZ={np.sum(y_test==1)}")

    # ── 2. Class weights (fix HC/SZ imbalance) ────────────────────────────────
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(cw))
    print(f"\nClass weights: HC={cw[0]:.3f}  SZ={cw[1]:.3f}")

    # ── 3. tf.data pipelines for sequence features ────────────────────────────
    def make_ds(X_, y_, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((X_, y_.astype(np.float32)))
        if shuffle:
            ds = ds.shuffle(len(X_), seed=SEED)
        return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(X_train, y_train, shuffle=True)
    val_ds   = make_ds(X_val,   y_val)
    test_ds  = make_ds(X_test,  y_test)

    # ── 4. Phase 1 — Train BiLSTM on sequence features ───────────────────────
    print("\n" + "=" * 55)
    print("Phase 1: Training BiLSTM Head on Sequence Features …")
    
    head_model = build_bilstm_head(input_shape=X_train.shape[1:])
    head_model.summary(line_length=90)

    # Compile with Adam
    head_model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    hist1 = head_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=get_callbacks(out_dir, "phase1"),
        verbose=1,
    )
    plot_history(hist1, out_dir, "phase1")
    active_model = head_model

    # ── 5. Phase 2 — Fine-tuning (Requires full images to fine-tune backbone) ──
    if args.fine_tune > 0:
        print("\n" + "=" * 55)
        print(f"Phase 2: Fine-tuning top {args.fine_tune} backbone layers on full images …")
        
        # Load images for training
        X_img, y_img, _ = load_dataset(data_dir, IMG_H, IMG_W)
        X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
            X_img, y_img, test_size=0.20, random_state=SEED, stratify=y_img
        )
        X_train_img, X_val_img, y_train_img, y_val_img = train_test_split(
            X_train_img, y_train_img, test_size=0.20, random_state=SEED, stratify=y_train_img
        )

        # Build full end-to-end model
        full_model, backbone = build_cnn_bilstm(IMG_H, IMG_W)
        
        # Transfer head weights (BiLSTM + Dense layers)
        print("Transferring trained BiLSTM head weights from Phase 1 ...")
        for layer in head_model.layers:
            if any(name in layer.name for name in ["bilstm", "bn_head", "fc1", "output"]):
                full_model.get_layer(layer.name).set_weights(layer.get_weights())

        # Unfreeze and compile
        unfreeze_top(backbone, args.fine_tune)
        
        # Cosine decay LR schedule for fine-tuning
        steps_per_epoch = len(X_train_img) // BATCH
        lr_ft = optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=LR / 10,
            first_decay_steps=steps_per_epoch * 5,
            t_mul=2.0, m_mul=0.9, alpha=1e-7,
        )
        
        full_model.compile(
            optimizer=optimizers.Adam(lr_ft),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # Data pipelines for images
        def make_img_ds(X_, y_, shuffle=False):
            ds = tf.data.Dataset.from_tensor_slices((X_, y_.astype(np.float32)))
            if shuffle:
                ds = ds.shuffle(len(X_), seed=SEED)
            return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

        train_img_ds = make_img_ds(X_train_img, y_train_img, shuffle=True)
        val_img_ds   = make_img_ds(X_val_img,   y_val_img)

        hist2 = full_model.fit(
            train_img_ds,
            validation_data=val_img_ds,
            epochs=EPOCHS // 2,
            class_weight=class_weight,
            callbacks=get_callbacks(out_dir, "phase2"),
            verbose=1,
        )
        plot_history(hist2, out_dir, "phase2")
        active_model = full_model

        # Free GPU memory
        del X_img, X_train_img, X_val_img, X_test_img
        import gc; gc.collect()

    # ── 6. Evaluate on test set ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Evaluating on test set …")
    
    if args.fine_tune > 0:
        # Full model uses test images
        X_img, y_img, _ = load_dataset(data_dir, IMG_H, IMG_W)
        _, X_test_img, _, y_test_img = train_test_split(
            X_img, y_img, test_size=0.20, random_state=SEED, stratify=y_img
        )
        test_img_ds = tf.data.Dataset.from_tensor_slices((X_test_img, y_test_img.astype(np.float32))).batch(BATCH)
        test_loss, test_acc = active_model.evaluate(test_img_ds, verbose=0)
        y_prob = active_model.predict(test_img_ds, verbose=0).ravel()
        y_test_final = y_test_img
    else:
        # Head model uses sequence features
        test_loss, test_acc = active_model.evaluate(test_ds, verbose=0)
        y_prob = active_model.predict(test_ds, verbose=0).ravel()
        y_test_final = y_test

    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")

    y_pred = (y_prob >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test_final, y_pred, target_names=class_names))

    auc = roc_auc_score(y_test_final, y_prob)
    print(f"  AUC-ROC : {auc:.4f}")

    # ── 7. Save plots & model ─────────────────────────────────────────────────
    plot_confusion(y_test_final, y_pred, class_names, out_dir)
    plot_roc(y_test_final, y_prob, out_dir)

    active_model.save(str(out_dir / "cnn_bilstm_eeg_final.keras"))
    print(f"\n✅ Model saved → {out_dir / 'cnn_bilstm_eeg_final.keras'}")

    # ── 8. Summary card ──────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print(f"  Model        : CNN + BiLSTM (EfficientNetB0 backbone)")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples  : {len(X_val)}")
    print(f"  Test samples : {len(X_test)}")
    print(f"  Test Acc     : {test_acc * 100:.2f}%")
    print(f"  AUC-ROC      : {auc:.4f}")
    print("━" * 50)
    print(f"All outputs → {out_dir}")


if __name__ == "__main__":
    main()
