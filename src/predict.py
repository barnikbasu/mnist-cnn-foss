#!/usr/bin/env python3

"""
Prediction script for the trained MNIST CNN model.

This script:
- Loads the trained CNN
- Loads the MNIST test set
- Selects reproducible test samples
- Displays predictions and confidence scores
- Generates a detailed prediction visualization

Usage:
    python src/predict.py

Prerequisite:
    Run 'python src/train.py' first to generate the trained model.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


SEED = 42
MODEL_PATH = "models/mnist_cnn.keras"
OUTPUT_PATH = "results/sample_predictions.png"


def load_model(model_path=MODEL_PATH):
    """Load the trained model from disk."""

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run 'python src/train.py' first.")
        raise FileNotFoundError(model_path)

    print(f"Loading model from {model_path}...")

    model = keras.models.load_model(model_path)

    print("Model loaded successfully.\n")

    return model


def load_test_data():
    """Load and preprocess the MNIST test set."""

    print("Loading MNIST test data...")

    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values.
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension.
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"Loaded {len(x_test)} test images.\n")

    return x_test, y_test


def make_predictions(
    model,
    x_test,
    y_test,
    num_predictions=5,
):
    """Make predictions on reproducibly selected test samples."""

    print(
        f"Making predictions on {num_predictions} "
        "test images...\n"
    )

    rng = np.random.default_rng(SEED)

    indices = rng.choice(
        len(x_test),
        size=num_predictions,
        replace=False,
    )

    predictions_data = []

    for number, test_idx in enumerate(indices, start=1):
        image = x_test[test_idx]
        actual_label = int(y_test[test_idx])

        prediction = model.predict(
            image[np.newaxis, ...],
            verbose=0,
        )

        predicted_label = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_label])

        is_correct = actual_label == predicted_label

        predictions_data.append(
            {
                "image": image,
                "actual": actual_label,
                "predicted": predicted_label,
                "confidence": confidence,
                "correct": is_correct,
                "probabilities": prediction[0],
            }
        )

        status = (
            "CORRECT"
            if is_correct
            else "INCORRECT"
        )

        print(f"Prediction {number}: {status}")
        print(f"  Actual:     {actual_label}")
        print(f"  Predicted:  {predicted_label}")
        print(f"  Confidence: {confidence:.2%}")
        print()

    return predictions_data


def visualize_predictions(
    predictions_data,
    output_path=OUTPUT_PATH,
):
    """Create a detailed visualization of predictions."""

    num_predictions = len(predictions_data)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    fig = plt.figure(
        figsize=(16, 4 * num_predictions)
    )

    for index, prediction in enumerate(
        predictions_data,
        start=1,
    ):
        # Image.
        ax1 = plt.subplot(
            num_predictions,
            2,
            2 * index - 1,
        )

        image = prediction["image"].reshape(28, 28)

        ax1.imshow(
            image,
            cmap="gray",
        )

        ax1.set_xticks([])
        ax1.set_yticks([])

        title_color = (
            "green"
            if prediction["correct"]
            else "red"
        )

        ax1.set_title(
            f"Actual: {prediction['actual']} | "
            f"Predicted: {prediction['predicted']}\n"
            f"Confidence: {prediction['confidence']:.2%}",
            fontsize=12,
            fontweight="bold",
            color=title_color,
        )

        # Probability distribution.
        ax2 = plt.subplot(
            num_predictions,
            2,
            2 * index,
        )

        probabilities = prediction["probabilities"]

        ax2.bar(
            range(10),
            probabilities,
            edgecolor="black",
            linewidth=1,
        )

        ax2.set_xlabel("Digit")
        ax2.set_ylabel("Probability")
        ax2.set_title("Prediction Probabilities")
        ax2.set_ylim([0, 1])
        ax2.grid(
            True,
            alpha=0.3,
            axis="y",
        )

        for digit, probability in enumerate(
            probabilities
        ):
            if probability > 0.05:
                ax2.text(
                    digit,
                    probability + 0.02,
                    f"{probability:.2%}",
                    ha="center",
                    fontsize=9,
                )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved detailed predictions visualization "
        f"to {output_path}"
    )


def print_statistics(predictions_data):
    """Print statistics for the selected predictions."""

    correct_count = sum(
        prediction["correct"]
        for prediction in predictions_data
    )

    total_count = len(predictions_data)

    accuracy = correct_count / total_count

    print("\n" + "=" * 60)
    print("PREDICTION STATISTICS")
    print("=" * 60)

    print(f"Total predictions: {total_count}")
    print(f"Correct: {correct_count}")
    print(
        f"Incorrect: "
        f"{total_count - correct_count}"
    )
    print(f"Sample accuracy: {accuracy:.2%}")

    print("=" * 60 + "\n")


def main():
    """Run the standalone prediction pipeline."""

    print("\n" + "=" * 60)
    print("MNIST CNN - Prediction Script")
    print("=" * 60 + "\n")

    model = load_model()

    x_test, y_test = load_test_data()

    predictions_data = make_predictions(
        model,
        x_test,
        y_test,
        num_predictions=5,
    )

    visualize_predictions(
        predictions_data
    )

    print_statistics(
        predictions_data
    )

    print("Predictions complete!")


if __name__ == "__main__":
    main()
