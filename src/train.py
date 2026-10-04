#!/usr/bin/env python3

"""
Training script for MNIST Handwritten Digit Recognition using a CNN.

This script:
- Downloads and preprocesses the MNIST dataset
- Builds a Convolutional Neural Network
- Trains the model with validation
- Evaluates the model on the test set
- Generates accuracy, loss, and prediction visualizations
- Saves the trained model and evaluation metrics

Usage:
    python src/train.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


SEED = 42
EPOCHS = 10
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.10


def set_seeds(seed=SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    print(f"Random seeds set to {seed} for reproducibility.")


def load_and_preprocess_data():
    """Load MNIST and normalize images to the [0, 1] range."""
    print("Loading MNIST dataset...")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print(f"Original training shape: {x_train.shape}")
    print(f"Original test shape:     {x_test.shape}")

    # Normalize pixel values from [0, 255] to [0, 1].
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add the channel dimension: (28, 28) -> (28, 28, 1).
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"Preprocessed training shape: {x_train.shape}")
    print(f"Preprocessed test shape:     {x_test.shape}")
    print(f"Pixel value range: [{x_train.min():.1f}, {x_train.max():.1f}]")
    print(f"Classes: {np.unique(y_train)}")

    return x_train, y_train, x_test, y_test


def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Build the CNN architecture.

    Architecture:
        Input: 28x28x1
            |
        Conv2D: 32 filters, 3x3, valid padding + ReLU
            |
        MaxPooling2D: 2x2
            |
        Conv2D: 64 filters, 3x3, valid padding + ReLU
            |
        MaxPooling2D: 2x2
            |
        Flatten
            |
        Dense: 128 + ReLU
            |
        Dropout: 0.5
            |
        Dense: 10 + Softmax
    """

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="valid",
            ),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="valid",
            ),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def compile_model(model):
    """Compile the CNN using Adam and sparse categorical crossentropy."""
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nModel compiled successfully.")
    return model


def train_model(
    model,
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
):
    """Train the CNN and return the training history."""

    print("\nTraining configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Validation split: {validation_split}")
    print("  - Optimizer: Adam")
    print("  - Loss: Sparse Categorical Crossentropy")

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )

    return history


def evaluate_model(model, x_test, y_test):
    """Evaluate the trained model on the held-out test set."""

    print("\nEvaluating on test set...")

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    return test_loss, test_accuracy


def plot_accuracy(history, output_path="results/accuracy.png"):
    """Plot training and validation accuracy."""

    plt.figure(figsize=(10, 6))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
        linewidth=2,
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2,
    )

    plt.title("Model Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved accuracy plot to {output_path}")


def plot_loss(history, output_path="results/loss.png"):
    """Plot training and validation loss."""

    plt.figure(figsize=(10, 6))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
        linewidth=2,
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
        linewidth=2,
    )

    plt.title("Model Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved loss plot to {output_path}")


def plot_predictions(
    model,
    x_test,
    y_test,
    num_samples=5,
    output_path="results/predictions.png",
):
    """Plot reproducible sample predictions with confidence scores."""

    rng = np.random.default_rng(SEED)

    indices = rng.choice(
        len(x_test),
        size=num_samples,
        replace=False,
    )

    fig, axes = plt.subplots(
        1,
        num_samples,
        figsize=(15, 3),
    )

    fig.suptitle(
        "Sample Predictions: Actual vs Predicted",
        fontsize=14,
        fontweight="bold",
    )

    for idx, ax in enumerate(axes):
        test_idx = indices[idx]

        image = x_test[test_idx]
        actual_label = y_test[test_idx]

        prediction = model.predict(
            x_test[test_idx : test_idx + 1],
            verbose=0,
        )

        predicted_label = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_label])

        ax.imshow(
            image.reshape(28, 28),
            cmap="gray",
        )

        ax.set_xticks([])
        ax.set_yticks([])

        title_color = (
            "green"
            if actual_label == predicted_label
            else "red"
        )

        ax.set_title(
            f"Actual: {actual_label}\n"
            f"Predicted: {predicted_label}\n"
            f"Confidence: {confidence:.2%}",
            color=title_color,
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved predictions plot to {output_path}")


def save_model(model, output_path="models/mnist_cnn.keras"):
    """Save the trained model."""

    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    model.save(output_path)

    print(f"Saved model to {output_path}")


def save_metrics(
    test_loss,
    test_accuracy,
    output_path="results/metrics.txt",
):
    """Save final evaluation metrics and training configuration."""

    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("MNIST CNN - Final Results\n")
        file.write("=" * 50 + "\n")
        file.write(
            f"Test Accuracy: {test_accuracy:.4f} "
            f"({test_accuracy * 100:.2f}%)\n"
        )
        file.write(f"Test Loss:     {test_loss:.4f}\n")
        file.write(f"Epochs: {EPOCHS}\n")
        file.write(f"Batch Size: {BATCH_SIZE}\n")
        file.write("Validation Split: 0.10\n")
        file.write("Optimizer: Adam\n")
        file.write(
            "Loss Function: Sparse Categorical Crossentropy\n"
        )
        file.write(f"Random Seed: {SEED}\n")

    print(f"Saved metrics to {output_path}")


def print_model_summary(model):
    """Print the model architecture."""

    print("\nModel Architecture:")
    print("=" * 60)

    model.summary()

    print("=" * 60)


def main():
    """Run the complete MNIST CNN training pipeline."""

    print("\n" + "=" * 60)
    print("MNIST Handwritten Digit Recognition - CNN Training")
    print("=" * 60 + "\n")

    # Reproducibility.
    set_seeds(SEED)

    # Load and preprocess data.
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Build model.
    print("\nBuilding model...")

    model = build_model(
        input_shape=(28, 28, 1),
        num_classes=10,
    )

    print_model_summary(model)

    # Compile.
    model = compile_model(model)

    # Train.
    history = train_model(
        model,
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
    )

    # Evaluate.
    test_loss, test_accuracy = evaluate_model(
        model,
        x_test,
        y_test,
    )

    # Create output directories.
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Generate visualizations.
    print("\nGenerating visualizations...")

    plot_accuracy(history)
    plot_loss(history)

    plot_predictions(
        model,
        x_test,
        y_test,
        num_samples=5,
    )

    # Save model and metrics.
    save_model(model)
    save_metrics(
        test_loss,
        test_accuracy,
    )

    # Final results.
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"Test Accuracy: {test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)"
    )

    print(f"Test Loss:     {test_loss:.4f}")

    print("=" * 60)
    print("\nTraining complete!")

    print("\nGenerated files:")
    print("  - models/mnist_cnn.keras")
    print("  - results/accuracy.png")
    print("  - results/loss.png")
    print("  - results/predictions.png")
    print("  - results/metrics.txt")

    print("\nNext step:")
    print("  Run 'python src/predict.py' for standalone predictions.")


if __name__ == "__main__":
    main()
