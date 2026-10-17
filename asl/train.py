"""EfficientNet-B0 image baseline; no training is claimed before actual data is supplied."""
import argparse
import json
from pathlib import Path
import tensorflow as tf
from .constants import IMAGE_SIZE

def build_model(classes, pretrained=True):
    inp = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.07),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.12),
    ])
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet" if pretrained else None,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base.trainable = False
    x = base(aug(inp), training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    out = tf.keras.layers.Dense(classes, activation="softmax")(x)
    network = tf.keras.Model(inp, out)
    network.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return network, base

def dataset(folder, batch, shuffle):
    return tf.keras.utils.image_dataset_from_directory(folder, image_size=(IMAGE_SIZE,IMAGE_SIZE), batch_size=batch, label_mode="int", shuffle=shuffle, seed=42)

def train(args):
    tf.keras.utils.set_random_seed(42)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    source = Path(args.data)
    train_data = dataset(source / "train", args.batch_size, True)
    val_data = dataset(source / "val", args.batch_size, False)
    if train_data.class_names != val_data.class_names:
        raise ValueError("Training and validation labels differ")
    labels = train_data.class_names
    if len(labels) < 2:
        raise ValueError("At least two labels required")
    (out / "labels.json").write_text(json.dumps({"classes":labels},indent=2))
    model, backbone = build_model(len(labels), not args.no_pretrained)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(out/"best.keras"),monitor="val_loss",save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)]
    histories = []
    histories.append(model.fit(train_data.prefetch(tf.data.AUTOTUNE), validation_data=val_data, epochs=args.epochs,callbacks=callbacks).history)
    if args.fine_tune_epochs:
        backbone.trainable = True
        for layer in backbone.layers[:-25]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer,tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        histories.append(model.fit(train_data,validation_data=val_data,epochs=args.fine_tune_epochs,callbacks=callbacks).history)
    (out/"history.json").write_text(json.dumps([{key:[float(x) for x in values] for key,values in history.items()} for history in histories],indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/split")
    p.add_argument("--out", default="outputs/static")
    p.add_argument("--epochs",type=int,default=15)
    p.add_argument("--fine-tune-epochs",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--no-pretrained", action="store_true")
    train(p.parse_args())
