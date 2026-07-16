from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

X_path = DATA_DIR / "X.npy"
y_path = DATA_DIR / "y.npy"


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

if not X_path.exists() or not y_path.exists():
    raise FileNotFoundError(
        "Could not find data/X.npy and data/y.npy. "
        "Please make sure your dataset files are inside the data folder."
    )

X = np.load(X_path).astype("float32")
y = np.load(y_path).astype("int64")

print("Loaded dataset")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label counts:", np.unique(y, return_counts=True))

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

if X.ndim != 4:
    raise ValueError(f"Expected X to be 4D, but got shape {X.shape}")

if X.shape[-1] not in [1, 3, 4, 5]:
    raise ValueError(
        f"Expected channels-last format: (N, height, width, channels). Got {X.shape}"
    )


# --------------------------------------------------
# 3. Train / validation / test split
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nSplit sizes")
print("Train:", X_train.shape, np.unique(y_train, return_counts=True))
print("Val:  ", X_val.shape, np.unique(y_val, return_counts=True))
print("Test: ", X_test.shape, np.unique(y_test, return_counts=True))


# --------------------------------------------------
# 4. Normalise using training set only
# --------------------------------------------------

mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-6

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

print("\nNormalisation complete")
print("Train mean:", float(X_train.mean()))
print("Train std:", float(X_train.std()))


# --------------------------------------------------
# 5. Build smaller CNN
# --------------------------------------------------

def build_smaller_cnn(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),

        layers.Dense(32, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
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


model = build_smaller_cnn(X_train.shape[1:])
model.summary()


# --------------------------------------------------
# 6. Class weights
# --------------------------------------------------

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

print("\nClass weights:", class_weight)


# --------------------------------------------------
# 7. Train model
# --------------------------------------------------

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(MODELS_DIR / "best_baseline_cnn.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)


# --------------------------------------------------
# 8. Tune threshold using validation data
# --------------------------------------------------

y_val_prob = model.predict(X_val).ravel()

print("\nValidation probability summary")
print("min:", float(y_val_prob.min()))
print("max:", float(y_val_prob.max()))
print("mean:", float(y_val_prob.mean()))

thresholds = np.linspace(0.01, 0.99, 99)

best_threshold = 0.5
best_f1 = 0.0

for threshold in thresholds:
    y_val_pred = (y_val_prob >= threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest threshold from validation set:", best_threshold)
print("Best validation F1:", best_f1)


# --------------------------------------------------
# 9. Evaluate on test set using best threshold
# --------------------------------------------------

print("\nEvaluating on test set")

test_results = model.evaluate(X_test, y_test, verbose=0)

for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")

y_prob = model.predict(X_test).ravel()

print("\nTest probability summary")
print("min:", float(y_prob.min()))
print("max:", float(y_prob.max()))
print("mean:", float(y_prob.mean()))

print("\nFirst 10 non-lens probabilities:")
print(y_prob[y_test == 0][:10])

print("\nFirst 10 lens probabilities:")
print(y_prob[y_test == 1][:10])

y_pred = (y_prob >= best_threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=["non_lens", "lens"],
    zero_division=0
)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(report)

print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)


# --------------------------------------------------
# 10. Save model, metrics and plots
# --------------------------------------------------

model.save(MODELS_DIR / "final_baseline_cnn.keras")

metrics = {
    "best_threshold": float(best_threshold),
    "best_validation_f1": float(best_f1),
    "test_loss": float(test_results[0]),
    "test_accuracy_default_metric": float(test_results[1]),
    "test_precision_default_metric": float(test_results[2]),
    "test_recall_default_metric": float(test_results[3]),
    "test_roc_auc_metric": float(test_results[4]),
    "test_pr_auc_metric": float(test_results[5]),
    "roc_auc_sklearn": float(roc_auc),
    "pr_auc_sklearn": float(pr_auc),
    "confusion_matrix_best_threshold": cm.tolist(),
}

with open(RESULTS_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open(RESULTS_DIR / "classification_report.txt", "w") as f:
    f.write(report)

plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and validation loss")
plt.savefig(RESULTS_DIR / "loss_curve.png", bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and validation accuracy")
plt.savefig(RESULTS_DIR / "accuracy_curve.png", bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(history.history["roc_auc"], label="train_roc_auc")
plt.plot(history.history["val_roc_auc"], label="val_roc_auc")
plt.xlabel("Epoch")
plt.ylabel("ROC-AUC")
plt.legend()
plt.title("Training and validation ROC-AUC")
plt.savefig(RESULTS_DIR / "roc_auc_curve.png", bbox_inches="tight")
plt.close()

print("\nSaved outputs:")
print("Results folder:", RESULTS_DIR)
print("Models folder:", MODELS_DIR)