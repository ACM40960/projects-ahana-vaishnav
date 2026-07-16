from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# 1. Load dataset
# -----------------------------
data_dir = Path("project_dataset")
X_path = data_dir / "X.npy"
y_path = data_dir / "y.npy"

if not X_path.exists() or not y_path.exists():
    raise FileNotFoundError(
        "Could not find project_dataset/X.npy and project_dataset/y.npy. "
        "Please create the binary dataset first."
    )

X = np.load(X_path).astype("float32")
y = np.load(y_path).astype("int64")

print("Loaded data")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", np.unique(y, return_counts=True))

# Expected shape: N, height, width, channels
if X.ndim != 4:
    raise ValueError(f"Expected X to have 4 dimensions, got shape {X.shape}")

if X.shape[-1] not in [1, 3, 4, 5]:
    raise ValueError(
        f"Expected channels-last format like (N, H, W, C), got {X.shape}. "
        "If your data is (N, C, H, W), transpose it before training."
    )

# Replace NaNs/Infs if present
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# -----------------------------
# 2. Train / validation / test split
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nSplit sizes")
print("Train:", X_train.shape, np.unique(y_train, return_counts=True))
print("Val:  ", X_val.shape, np.unique(y_val, return_counts=True))
print("Test: ", X_test.shape, np.unique(y_test, return_counts=True))


# -----------------------------
# 3. Normalise using training set only
# -----------------------------
mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-6

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

print("\nAfter normalisation")
print("Train mean approx:", float(X_train.mean()))
print("Train std approx:", float(X_train.std()))


# -----------------------------
# 4. Build baseline CNN
# -----------------------------
def build_baseline_cnn(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.15),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.20),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.40),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
        ]
    )

    return model


model = build_baseline_cnn(X_train.shape[1:])
model.summary()


# -----------------------------
# 5. Class weights
# -----------------------------
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
print("\nClass weights:", class_weight)


# -----------------------------
# 6. Train model
# -----------------------------
outdir = Path("cnn_results")
outdir.mkdir(exist_ok=True)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(outdir / "best_baseline_cnn.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)


# -----------------------------
# 7. Evaluate on test set
# -----------------------------
print("\nEvaluating on test set")
test_results = model.evaluate(X_test, y_test, verbose=0)
for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")

y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["non_lens", "lens"])

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(report)

print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)


# -----------------------------
# 8. Save outputs
# -----------------------------
model.save(outdir / "final_baseline_cnn.keras")

metrics = {
    "test_loss": float(test_results[0]),
    "test_accuracy": float(test_results[1]),
    "test_precision": float(test_results[2]),
    "test_recall": float(test_results[3]),
    "test_roc_auc_metric": float(test_results[4]),
    "test_pr_auc_metric": float(test_results[5]),
    "roc_auc_sklearn": float(roc_auc),
    "pr_auc_sklearn": float(pr_auc),
    "confusion_matrix": cm.tolist(),
}

with open(outdir / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open(outdir / "classification_report.txt", "w") as f:
    f.write(report)

# Loss plot
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and validation loss")
plt.savefig(outdir / "loss_curve.png", bbox_inches="tight")
plt.close()

# Accuracy plot
plt.figure()
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and validation accuracy")
plt.savefig(outdir / "accuracy_curve.png", bbox_inches="tight")
plt.close()

# ROC-AUC plot from training history
plt.figure()
plt.plot(history.history["roc_auc"], label="train_roc_auc")
plt.plot(history.history["val_roc_auc"], label="val_roc_auc")
plt.xlabel("Epoch")
plt.ylabel("ROC-AUC")
plt.legend()
plt.title("Training and validation ROC-AUC")
plt.savefig(outdir / "roc_auc_curve.png", bbox_inches="tight")
plt.close()

print("\nSaved results in:", outdir)
print("Best model:", outdir / "best_baseline_cnn.keras")
print("Final model:", outdir / "final_baseline_cnn.keras")

print("Predicted probabilities:")
print("min:", y_prob.min())
print("max:", y_prob.max())
print("mean:", y_prob.mean())

print("Non-lens probabilities:")
print(y_prob[y_test == 0][:20])

print("Lens probabilities:")
print(y_prob[y_test == 1][:20])
