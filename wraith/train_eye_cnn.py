"""
Tiny CNN for eye open/closed classification.

This script downloads the CEW dataset (or uses an existing dataset),
builds a small CNN, trains it, evaluates it, and exports to TF.js.

Output:
  - Saved model: ./model_export/saved_model
  - TF.js model: ./wraith/model/eye_state_model (model.json + shards)

Usage (Linux/macOS):
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python train_eye_cnn.py --epochs 8 --img_w 48 --img_h 24

Notes:
  - If CEW link changes, place your dataset under ./data/eyes with
    subfolders open/ and closed/ containing images of single eyes.
  - You can generate eye crops from face images using MediaPipe offline if needed.
"""
import argparse
import os
import sys
import shutil
from pathlib import Path
import zipfile
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow import keras  # type: ignore[reportUnknownVariableType]

try:
    import tensorflowjs as tfjs
except Exception:
    tfjs = None


def download_cew_if_needed(dst_dir: Path):
    # CEW (Closed Eyes in the Wild): academic dataset; URLs sometimes change.
    # We avoid hardcoding a brittle URL. If not available, we skip downloading.
    # User can place images under data/eyes/open and data/eyes/closed.
    print("If you have CEW or another dataset, place it under data/eyes/{open,closed}.")
    print("Skipping auto-download.")


def build_dataset(root: Path, img_w: int, img_h: int, batch: int = 64):
    root = Path(root)
    open_dir = root / "open"
    closed_dir = root / "closed"
    if not open_dir.exists() or not closed_dir.exists():
        raise FileNotFoundError(f"Expected dataset at {root}/open and {root}/closed")

    ds_train = tf.keras.utils.image_dataset_from_directory(  # type: ignore[attr-defined]
        str(root),
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=(img_h, img_w),
        batch_size=batch,
        validation_split=0.2,
        subset="training",
        seed=42,
    )
    ds_val = tf.keras.utils.image_dataset_from_directory(  # type: ignore[attr-defined]
        str(root),
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=(img_h, img_w),
        batch_size=batch,
        validation_split=0.2,
        subset="validation",
        seed=42,
    )
    # Normalize to [0,1]
    def norm(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y
    autotune = tf.data.AUTOTUNE
    ds_train = ds_train.map(norm, num_parallel_calls=autotune).cache().shuffle(2048).prefetch(autotune)
    ds_val = ds_val.map(norm, num_parallel_calls=autotune).cache().prefetch(autotune)
    return ds_train, ds_val


def build_model(img_w: int, img_h: int) -> keras.Model:
    inputs = keras.Input(shape=(img_h, img_w, 1))
    x = inputs
    x = keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(48, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs, name='eye_state_cnn')
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--img_w', type=int, default=48)
    p.add_argument('--img_h', type=int, default=24)
    p.add_argument('--data_dir', type=str, default='data/eyes')
    p.add_argument('--export_dir', type=str, default='model_export')
    p.add_argument('--tfjs_out', type=str, default='wraith/model/eye_state_model')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Dataset dir {data_dir} not found. Creating placeholder.")
        data_dir.mkdir(parents=True, exist_ok=True)
        print("Please add images to data/eyes/open and data/eyes/closed and rerun.")
        sys.exit(1)

    ds_train, ds_val = build_dataset(data_dir, args.img_w, args.img_h, args.batch)
    model = build_model(args.img_w, args.img_h)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
    ]
    model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, callbacks=callbacks)

    eval_res = model.evaluate(ds_val, verbose=0)
    print({k: float(v) for k, v in zip(model.metrics_names, eval_res)})

    export_dir = Path(args.export_dir)
    saved_dir = export_dir / 'saved_model'
    export_dir.mkdir(parents=True, exist_ok=True)
    # Keras 3: use model.export() for SavedModel format
    try:
        model.export(str(saved_dir))
        print(f"SavedModel exported to {saved_dir}")
    except Exception as e:
        print(f"SavedModel export failed ({e}); saving native Keras file instead.")
        keras_path = export_dir / 'eye_state_cnn.keras'
        model.save(keras_path)
        print(f"Keras model saved to {keras_path}")

    if tfjs is None:
        print("tensorflowjs is not installed; skipping TF.js export.\nInstall with: pip install tensorflowjs")
        return

    tfjs_out = Path(args.tfjs_out)
    tfjs_out.mkdir(parents=True, exist_ok=True)
    tfjs.converters.save_keras_model(model, str(tfjs_out))
    print(f"TF.js model exported to {tfjs_out}")


if __name__ == '__main__':
    main()
