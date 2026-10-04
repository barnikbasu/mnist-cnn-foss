#!/usr/bin/env python3

"""
Training script for MNIST Handwritten Digit Recognition CNN.

This script:
- Downloads and preprocesses the MNIST dataset
- Builds a Convolutional Neural Network
- Trains the model with validation
- Evaluates on test set
- Generates performance visualizations
- Saves the trained model

Usage:
    python src/train.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seeds set to {seed} for reproducibility")


def load_and_preprocess_data():
    """Load MNIST dataset and preprocess."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"Original shapes: {x_train.shape}, {y_train.shape}")
    
    # Normalize pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension: (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    print(f"Preprocessed shapes: {x_train.shape}, {y_train.shape}")
    print(f"Pixel value range: [{x_train.min()}, {x_train.max()}]")
    print(f"Classes: {np.unique(y_train)}")
    
    return x_train, y_train, x_test, y_test


def build_model(input_shape=(28, 28, 1), num_classes=10):
    """Build CNN architecture.
    
    Architecture:
        Input (28×28×1)
            ↓
        Conv2D(32, 3×3) + ReLU
            ↓
        MaxPooling(2×2)
            ↓
        Conv2D(64, 3×3) + ReLU
            ↓
        MaxPooling(2×2)
            ↓
        Flatten
            ↓
        Dense(128) + ReLU + Dropout(0.5)
            ↓
        Dense(10) + Softmax
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model):
    """Compile the model."""
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("\nModel compiled successfully.")
    return model


def train_model(model, x_train, y_train, epochs=10, batch_size=128, validation_split=0.1):
    """Train the model."""
    print(f"\nTraining configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Validation split: {validation_split}")
    print(f"  - Optimizer: Adam")
    print(f"  - Loss: Sparse Categorical Crossentropy\n")
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    
    return history


def evaluate_model(model, x_test, y_test):
    """Evaluate model on test set."""
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    
    return test_loss, test_accuracy


def plot_accuracy(history, output_path='results/accuracy.png'):
    """Plot training vs validation accuracy."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved accuracy plot to {output_path}")
    plt.close()


def plot_loss(history, output_path='results/loss.png'):
    """Plot training vs validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved loss plot to {output_path}")
    plt.close()


def plot_predictions(model, x_test, y_test, num_samples=5, output_path='results/predictions.png'):
    """Plot sample predictions with confidence scores."""
    # Select random test samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle('Sample Predictions (Actual vs Predicted)', fontsize=14, fontweight='bold')
    
    for idx, ax in enumerate(axes):
        test_idx = indices[idx]
        image = x_test[test_idx].reshape(28, 28)
        actual_label = y_test[test_idx]
        
        # Make prediction
        prediction = model.predict(x_test[test_idx:test_idx+1], verbose=0)
        predicted_label = np.argmax(prediction[0])
        confidence = prediction[0][predicted_label]
        
        # Plot image
        ax.imshow(image, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Color: green if correct, red if incorrect
        color = 'green' if actual_label == predicted_label else 'red'
        ax.set_title(
            f'Actual: {actual_label}\nPredicted: {predicted_label}\nConfidence: {confidence:.2%}',
            color=color,
            fontsize=10,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved predictions plot to {output_path}")
    plt.close()


def save_model(model, output_path='models/mnist_cnn.keras'):
    """Save trained model."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"Saved model to {output_path}")


def save_metrics(test_loss, test_accuracy, output_path='results/metrics.txt'):
    """Save evaluation metrics to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("MNIST CNN - Final Results\n")
        f.write("="*50 + "\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)\n")
        f.write(f"Test Loss:     {test_loss:.4f}\n")
    print(f"Saved metrics to {output_path}")


def print_model_summary(model):
    """Print model architecture summary."""
    print("\nModel Architecture:")
    print("="*60)
    model.summary()
    print("="*60)


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("MNIST Handwritten Digit Recognition - CNN Training")
    print("="*60 + "\n")
    
    # Set random seeds for reproducibility
    set_seeds(42)
    
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # Build model
    print("\nBuilding model...")
    model = build_model(input_shape=(28, 28, 1), num_classes=10)
    print_model_summary(model)
    
    # Compile model
    model = compile_model(model)
    
    # Train model
    history = train_model(model, x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)
    
    # Evaluate model
    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_accuracy(history)
    plot_loss(history)
    plot_predictions(model, x_test, y_test, num_samples=5)
    
    # Save model
    save_model(model)
    
    # Save metrics
    save_metrics(test_loss, test_accuracy)
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Test Loss:     {test_loss:.4f}")
    print("="*60 + "\n")
    print("✓ Training complete!")
    print("✓ Model saved to: models/mnist_cnn.keras")
    print("✓ Results saved to: results/")
    print("\nNext steps:")
    print("  1. Check results/metrics.txt for exact metrics")
    print("  2. View results/accuracy.png and results/loss.png")
    print("  3. Run 'python src/predict.py' for sample predictions")
    print()


if __name__ == '__main__':
    main()
