"""
Train a small CNN for mouth classification (neutral/open/smile/yawn).

Usage:
  python wraith/train_mouth_classifier.py --epochs 12 --img_w 64 --img_h 64

Dataset layout:
  data/yawn/neutral/
  data/yawn/open/
  data/yawn/smile/
  data/yawn/yawn/

Exports TF.js model to `wraith/model/mouth_classifier_model/`.
"""
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras  # type: ignore[reportUnknownVariableType]

try:
    import tensorflowjs as tfjs
except Exception:
    tfjs = None


def build_dataset(root: Path, img_w: int, img_h: int, batch: int = 64):
    ds_train_raw = keras.utils.image_dataset_from_directory(  # type: ignore[attr-defined]
        str(root),
        labels='inferred',
        label_mode='int',
        image_size=(img_h, img_w),
        batch_size=batch,
        validation_split=0.2,
        subset='training',
        seed=42,
    )
    ds_val_raw = keras.utils.image_dataset_from_directory(  # type: ignore[attr-defined]
        str(root),
        labels='inferred',
        label_mode='int',
        image_size=(img_h, img_w),
        batch_size=batch,
        validation_split=0.2,
        subset='validation',
        seed=42,
    )
    # capture class names before applying dataset transformations which strip attributes
    class_names = ds_train_raw.class_names  # type: ignore[attr-defined]

    def norm(x,y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y
    autotune = tf.data.AUTOTUNE
    ds_train = ds_train_raw.map(norm, num_parallel_calls=autotune).cache().shuffle(2048).prefetch(autotune)  # type: ignore[reportUnknownMemberType]
    ds_val = ds_val_raw.map(norm, num_parallel_calls=autotune).cache().prefetch(autotune)  # type: ignore[reportUnknownMemberType]
    return ds_train, ds_val, class_names


def build_model(img_w: int, img_h: int, n_classes: int):
    inputs = keras.Input(shape=(img_h, img_w, 3))
    x = inputs
    x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(48, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name='mouth_classifier')
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img_w', type=int, default=64)
    p.add_argument('--img_h', type=int, default=64)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--data_dir', type=str, default='data/mouth')
    p.add_argument('--tfjs_out', type=str, default='wraith/model/mouth_classifier_model')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print('Expected dataset at', data_dir)
        return

    ds_train, ds_val, class_names = build_dataset(data_dir, args.img_w, args.img_h, args.batch)
    n_classes = len(class_names)
    print('Classes:', class_names)

    model = build_model(args.img_w, args.img_h, n_classes)
    model.summary()
    model.fit(ds_train, validation_data=ds_val, epochs=args.epochs)

    export_dir = Path('model_export') / 'mouth_classifier_saved'
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.export(str(export_dir))
        print('SavedModel exported to', export_dir)
    except Exception as e:
        print('SavedModel export failed:', e)
        model.save(export_dir.with_suffix('.keras'))

    if tfjs is None:
        print('tensorflowjs not installed; skipping TF.js export')
        return
    tfjs_out = Path(args.tfjs_out)
    tfjs_out.mkdir(parents=True, exist_ok=True)
    tfjs.converters.save_keras_model(model, str(tfjs_out))
    print('TF.js model exported to', tfjs_out)


if __name__ == '__main__':
    main()
