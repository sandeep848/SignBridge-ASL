"""Isolated-sign temporal landmark model and reproducible local training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

FRAMES = 32
JOINTS = 21
FEATURES = 2 * JOINTS * 3 + 2

def resample_sequence(points: np.ndarray, frames: int = FRAMES) -> np.ndarray:
    """Resample by nearest frames; do not interpolate through missing hand detections."""
    p = np.asarray(points, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != FEATURES or len(p) < 1 or not np.isfinite(p).all():
        raise ValueError(f"Expected finite, nonempty sequence with shape (T,{FEATURES})")
    if frames < 1:
        raise ValueError("frames must be positive")
    indices = np.rint(np.linspace(0, len(p) - 1, frames)).astype(int)
    return p[indices]

def model(num_classes: int, frames: int = FRAMES):
    import tensorflow as tf
    if num_classes < 2:
        raise ValueError("At least two classes required")
    inp = tf.keras.Input((frames, FEATURES), name="landmark_sequence")
    x = tf.keras.layers.LayerNormalization()(inp)
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="gelu")(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="gelu")(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(96, return_sequences=True))(x)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, output, name="SignBridgeTemporal")

def load_partition(path: Path, label_map: dict):
    xs, ys = [], []
    for file in sorted(path.glob("*.npz")):
        with np.load(file, allow_pickle=False) as item:
            label = str(item["label"].item())
            if label not in label_map:
                raise ValueError(f"Unexpected label {label} in {file}")
            xs.append(resample_sequence(item["points"]))
            ys.append(label_map[label])
    if not xs:
        raise ValueError(f"No data in {path}")
    return np.stack(xs), np.asarray(ys, dtype=np.int32)

def train(args):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    root = Path(args.data)
    labels = []
    for path in sorted((root / "train").glob("*.npz")):
        with np.load(path, allow_pickle=False) as item:
            labels.append(str(item["label"].item()))
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError("At least two training classes are needed")
    lookup = {name: i for i, name in enumerate(classes)}
    train_x, train_y = load_partition(root / "train", lookup)
    val_x, val_y = load_partition(root / "val", lookup)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(json.dumps({"classes": classes}, indent=2))
    network = model(len(classes))
    network.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(out / "best.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    ]
    hist = network.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
    (out / "history.json").write_text(json.dumps({k: [float(v) for v in values] for k, values in hist.history.items()}, indent=2))
    (out / "run.json").write_text(json.dumps(vars(args), indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="outputs/temporal")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())

if __name__ == "__main__":
    main()
