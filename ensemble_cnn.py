"""
ensemble_cnn.py
Ensemble of CNNs for Fashion-MNIST classification, compared against a
single CNN model trained on the same data.

Pipeline (per task spec):
    1. Load Fashion-MNIST, keep only the first 50 training and 50 test
       records, normalize to [0, 1], reshape to (28, 28, 1).
    2. Split the 50 training records into train/validation sets.
    3. Define a small CNN: Conv2D(32,3x3,relu) -> MaxPool(2x2) -> Flatten
       -> Dense(10, softmax).
    4. Train an ensemble of 5 CNNs, each on a bootstrap resample (sampling
       with replacement) of the training set, for 3 epochs each.
    5. Evaluate the ensemble by averaging predicted probabilities across
       all 5 models, on both validation and test sets.
    6. Train a single CNN on the full (50-record) training set for
       comparison.
    7. Report and save accuracy for both approaches.
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import data_loader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_RECORDS = 50           # "first 50 records for training and testing"
N_ENSEMBLE_MODELS = 5
EPOCHS = 3
BATCH_SIZE = 8
RANDOM_STATE = 42
OUTPUT_DIR = "output"

CLASS_NAMES = data_loader.CLASS_NAMES

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# 1. Data loading & preparation
# ---------------------------------------------------------------------------
def load_and_prepare_data():
    (x_train_full, y_train_full), (x_test_full, y_test_full) = data_loader.load_data()

    # Select the first 50 records for training and testing.
    x_train = x_train_full[:N_RECORDS]
    y_train = y_train_full[:N_RECORDS]
    x_test = x_test_full[:N_RECORDS]
    y_test = y_test_full[:N_RECORDS]

    # Normalize to [0, 1].
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape to (28, 28, 1) for CNN input.
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Split training data into train/validation sets.
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    return x_tr, x_val, y_tr, y_val, x_test, y_test


# ---------------------------------------------------------------------------
# 2. CNN model definition
# ---------------------------------------------------------------------------
def build_cnn_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 3. Ensemble training (bootstrap resampling)
# ---------------------------------------------------------------------------
def train_ensemble(x_tr, y_tr, n_models=N_ENSEMBLE_MODELS, epochs=EPOCHS):
    models = []
    histories = []
    rng = np.random.RandomState(RANDOM_STATE)

    for i in range(n_models):
        print(f"\n--- Training ensemble model {i + 1}/{n_models} ---")
        # Bootstrap sample: sample the training data with replacement.
        indices = rng.choice(len(x_tr), size=len(x_tr), replace=True)
        x_boot, y_boot = x_tr[indices], y_tr[indices]

        model = build_cnn_model()
        history = model.fit(
            x_boot, y_boot,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        final_acc = history.history["accuracy"][-1]
        print(f"Model {i + 1} final training accuracy: {final_acc:.4f}")

        models.append(model)
        histories.append(history.history)

    return models, histories


def ensemble_predict_proba(models, x):
    """Average predicted class probabilities across all ensemble models."""
    all_probs = np.array([model.predict(x, verbose=0) for model in models])
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs


# ---------------------------------------------------------------------------
# 4. Single model training (for comparison)
# ---------------------------------------------------------------------------
def train_single_model(x_tr, y_tr, epochs=EPOCHS):
    print("\n--- Training single CNN model (full training set) ---")
    model = build_cnn_model()
    history = model.fit(
        x_tr, y_tr,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    final_acc = history.history["accuracy"][-1]
    print(f"Single model final training accuracy: {final_acc:.4f}")
    return model, history.history


# ---------------------------------------------------------------------------
# 5. Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_predictions(y_true, y_pred, name, split_name):
    acc = accuracy_score(y_true, y_pred)
    print(f"{name} — {split_name} accuracy: {acc:.4f}")
    return acc


def save_confusion_matrix(y_true, y_pred, title, out_path):
    labels_present = sorted(set(y_true) | set(y_pred))
    display_labels = [CLASS_NAMES[i] for i in labels_present]
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def save_sample_predictions(x, y_true, y_pred, title, out_path, n=10):
    n = min(n, len(x))
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        img = x[i].reshape(28, 28)
        pred_label = CLASS_NAMES[y_pred[i]]
        true_label = CLASS_NAMES[y_true[i]]
        correct = pred_label == true_label

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"P: {pred_label}\nT: {true_label}",
            color="green" if correct else "red",
            fontsize=8,
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def save_accuracy_comparison_chart(results, out_path):
    labels = ["Validation", "Test"]
    ensemble_scores = [results["ensemble"]["val_accuracy"], results["ensemble"]["test_accuracy"]]
    single_scores = [results["single"]["val_accuracy"], results["single"]["test_accuracy"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    bars1 = ax.bar(x - width / 2, ensemble_scores, width, label="Ensemble (5 CNNs)")
    bars2 = ax.bar(x + width / 2, single_scores, width, label="Single CNN")

    ax.set_ylabel("Accuracy")
    ax.set_title("Ensemble vs. Single Model Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading and preparing Fashion-MNIST data...")
    x_tr, x_val, y_tr, y_val, x_test, y_test = load_and_prepare_data()
    print(f"Train: {x_tr.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    # --- Train ensemble ---
    ensemble_models, ensemble_histories = train_ensemble(x_tr, y_tr)

    # --- Ensemble evaluation ---
    val_probs = ensemble_predict_proba(ensemble_models, x_val)
    test_probs = ensemble_predict_proba(ensemble_models, x_test)
    val_pred_ensemble = np.argmax(val_probs, axis=1)
    test_pred_ensemble = np.argmax(test_probs, axis=1)

    ensemble_val_acc = evaluate_predictions(y_val, val_pred_ensemble, "Ensemble", "validation")
    ensemble_test_acc = evaluate_predictions(y_test, test_pred_ensemble, "Ensemble", "test")

    # --- Train & evaluate single model ---
    single_model, single_history = train_single_model(x_tr, y_tr)

    val_probs_single = single_model.predict(x_val, verbose=0)
    test_probs_single = single_model.predict(x_test, verbose=0)
    val_pred_single = np.argmax(val_probs_single, axis=1)
    test_pred_single = np.argmax(test_probs_single, axis=1)

    single_val_acc = evaluate_predictions(y_val, val_pred_single, "Single model", "validation")
    single_test_acc = evaluate_predictions(y_test, test_pred_single, "Single model", "test")

    # --- Save visual outputs ---
    save_confusion_matrix(
        y_test, test_pred_ensemble,
        f"Ensemble Confusion Matrix - Test (acc={ensemble_test_acc:.2f})",
        os.path.join(OUTPUT_DIR, "confusion_matrix_ensemble_test.png"),
    )
    save_confusion_matrix(
        y_test, test_pred_single,
        f"Single Model Confusion Matrix - Test (acc={single_test_acc:.2f})",
        os.path.join(OUTPUT_DIR, "confusion_matrix_single_test.png"),
    )
    save_sample_predictions(
        x_test, y_test, test_pred_ensemble,
        "Ensemble: Sample Test Predictions (green=correct, red=wrong)",
        os.path.join(OUTPUT_DIR, "sample_predictions_ensemble.png"),
    )
    save_sample_predictions(
        x_test, y_test, test_pred_single,
        "Single Model: Sample Test Predictions (green=correct, red=wrong)",
        os.path.join(OUTPUT_DIR, "sample_predictions_single.png"),
    )

    # --- Results summary ---
    results = {
        "ensemble": {
            "val_accuracy": float(ensemble_val_acc),
            "test_accuracy": float(ensemble_test_acc),
        },
        "single": {
            "val_accuracy": float(single_val_acc),
            "test_accuracy": float(single_test_acc),
        },
        "config": {
            "n_records": N_RECORDS,
            "n_ensemble_models": N_ENSEMBLE_MODELS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "train_size": int(len(x_tr)),
            "val_size": int(len(x_val)),
            "test_size": int(len(x_test)),
        },
    }

    save_accuracy_comparison_chart(results, os.path.join(OUTPUT_DIR, "accuracy_comparison.png"))

    with open(os.path.join(OUTPUT_DIR, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "results_summary.txt"), "w") as f:
        f.write("Ensemble CNN vs. Single CNN — Fashion-MNIST Results\n")
        f.write("====================================================\n\n")
        f.write(f"Train records: {len(x_tr)} | Validation records: {len(x_val)} | Test records: {len(x_test)}\n")
        f.write(f"Ensemble size: {N_ENSEMBLE_MODELS} models | Epochs per model: {EPOCHS}\n\n")
        f.write(f"Ensemble  — Validation accuracy: {ensemble_val_acc:.4f} | Test accuracy: {ensemble_test_acc:.4f}\n")
        f.write(f"Single    — Validation accuracy: {single_val_acc:.4f} | Test accuracy: {single_test_acc:.4f}\n\n")

        val_diff = ensemble_val_acc - single_val_acc
        test_diff = ensemble_test_acc - single_test_acc
        f.write(f"Validation accuracy difference (ensemble - single): {val_diff:+.4f}\n")
        f.write(f"Test accuracy difference (ensemble - single): {test_diff:+.4f}\n")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Ensemble  — Val: {ensemble_val_acc:.4f} | Test: {ensemble_test_acc:.4f}")
    print(f"Single    — Val: {single_val_acc:.4f} | Test: {single_test_acc:.4f}")
    print("=" * 60)
    print(f"\nAll outputs saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
