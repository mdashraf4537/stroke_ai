"""
Medical Image Fine-Tuning Pipeline for Stroke Classification
Supports MRI (Haemorrhagic / Ischemic / Normal) and CT scans.
Implements:
1. Two-stage transfer learning (Head Warmup -> Deep Conv Fine-Tuning)
2. Balanced class weighting to solve the ischemic data scarcity problem
3. Model-specific ImageNet preprocessing (avoids double-rescaling bug)
4. Radiology-safe augmentations (horizontal flip, slight rotation/contrast jitter)
5. Automated validation evaluation & model export directly to models/
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Suppress verbose TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_dataset(data_dir, img_size=(224, 224), batch_size=16, val_split=0.2, seed=42):
    """Loads and splits the image dataset using stratified seeds."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


def calculate_class_weights(dataset):
    """Computes balanced class weights across the dataset."""
    y_all = []
    for _, labels in dataset:
        y_all.extend(labels.numpy())
    y_all = np.array(y_all)
    classes = np.unique(y_all)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_all)
    return dict(zip(classes, weights))


def build_stroke_model(num_classes, img_size=(224, 224), architecture="MobileNetV2"):
    """
    Builds a transfer learning model with proper preprocessing and dropout regularizers.
    """
    inputs = tf.keras.Input(shape=(*img_size, 3), name="scan_input")

    # Safe medical augmentations
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.04)(x)  # ~14 degrees max tilt
    x = tf.keras.layers.RandomBrightness(0.08)(x)
    x = tf.keras.layers.RandomContrast(0.08)(x)

    # Official backbone preprocessing
    if architecture == "MobileNetV2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_tensor=x
        )
    elif architecture == "DenseNet121":
        x = tf.keras.applications.densenet.preprocess_input(x)
        base_model = tf.keras.applications.DenseNet121(
            include_top=False, weights="imagenet", input_tensor=x
        )
    elif architecture == "ResNet50":
        x = tf.keras.applications.resnet50.preprocess_input(x)
        base_model = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_tensor=x
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Freeze backbone initially for stage 1 warmup
    base_model.trainable = False

    # Modern classification head with regularizers
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"Stroke_{architecture}")
    return model, base_model


def train_and_finetune(
    data_dir,
    output_model_path,
    architecture="MobileNetV2",
    stage1_epochs=6,
    stage2_epochs=12,
    unfreeze_layers=35,
    batch_size=16,
    img_size=(224, 224)
):
    print("=" * 70)
    print(f"Starting Fine-Tuning Pipeline: {architecture}")
    print(f"Dataset: {data_dir}")
    print(f"Output:  {output_model_path}")
    print("=" * 70)

    train_ds, val_ds, class_names = get_dataset(data_dir, img_size=img_size, batch_size=batch_size)
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    class_weights = calculate_class_weights(train_ds)
    print("Computed balanced class weights:", {class_names[k]: round(v, 3) for k, v in class_weights.items()})

    # Cache & prefetch for speed
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    model, base_model = build_stroke_model(num_classes, img_size=img_size, architecture=architecture)

    # -------------------------------------------------------------
    # STAGE 1: Train Head Only (Warmup)
    # -------------------------------------------------------------
    print("\n--- STAGE 1: Warmup Classifier Head (Backbone Frozen) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        class_weight=class_weights,
        verbose=1
    )

    # -------------------------------------------------------------
    # STAGE 2: Deep Fine-Tuning (Unfreeze Top Conv Layers)
    # -------------------------------------------------------------
    print(f"\n--- STAGE 2: Fine-Tuning Top {unfreeze_layers} Layers with Low Learning Rate ---")
    base_model.trainable = True
    
    # Freeze all layers except the last `unfreeze_layers`
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    
    trainable_count = sum(np.prod(v.shape) for v in model.trainable_weights)
    print(f"Trainable parameters in Stage 2: {trainable_count:,}")

    # Use smaller learning rate for fine-tuning with ReduceLROnPlateau
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # -------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL VALIDATION EVALUATION:")
    print("=" * 70)

    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Save model
    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
    model.save(output_model_path)
    print(f"\n[OK] Best fine-tuned model saved successfully to: {output_model_path}")
    return model, report


if __name__ == "__main__":
    default_mri_dir = r"C:\Users\ASUS\Downloads\archive\dataset\Stroke_classification"
    target_mri_path = os.path.join(os.path.dirname(__file__), "models", "stroke_cnn_model.keras")

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = default_mri_dir

    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    else:
        out_path = target_mri_path

    arch = sys.argv[3] if len(sys.argv) > 3 else "MobileNetV2"

    train_and_finetune(
        data_dir=data_dir,
        output_model_path=out_path,
        architecture=arch,
        stage1_epochs=5,
        stage2_epochs=12,
        unfreeze_layers=30
    )
