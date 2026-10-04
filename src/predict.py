#!/usr/bin/env python3

"""
Prediction script for trained MNIST CNN model.

This script loads a trained model and makes predictions on random test images,
displaying the actual label, predicted label, and confidence score.

Usage:
    python src/predict.py

Prerequisites:
    - Must run 'python src/train.py' first to generate the model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os


def load_model(model_path='models/mnist_cnn.keras'):
    """Load trained model from disk."""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run 'python src/train.py' first to train the model.")
        exit(1)
    
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!\n")
    return model


def load_test_data():
    """Load and preprocess MNIST test data."""
    print("Loading MNIST test data...")
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_test = np.expand_dims(x_test, axis=-1)
    
    print(f"Loaded {len(x_test)} test images\n")
    return x_test, y_test


def make_predictions(model, x_test, y_test, num_predictions=5):
    """Make predictions on random test samples."""
    print(f"Making predictions on {num_predictions} random test images...\n")
    
    # Select random indices
    indices = np.random.choice(len(x_test), num_predictions, replace=False)
    
    predictions_data = []
    
    for i, test_idx in enumerate(indices, 1):
        image = x_test[test_idx]
        actual_label = y_test[test_idx]
        
        # Make prediction
        prediction = model.predict(image[np.newaxis, ...], verbose=0)
        predicted_label = np.argmax(prediction[0])
        confidence = prediction[0][predicted_label]
        
        is_correct = actual_label == predicted_label
        
        predictions_data.append({
            'image': image,
            'actual': actual_label,
            'predicted': predicted_label,
            'confidence': confidence,
            'correct': is_correct,
            'probabilities': prediction[0]
        })
        
        # Print prediction details
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        print(f"Prediction {i}: {status}")
        print(f"  Actual:     {actual_label}")
        print(f"  Predicted:  {predicted_label}")
        print(f"  Confidence: {confidence:.2%}")
        print()
    
    return predictions_data


def visualize_predictions(predictions_data, output_path='results/sample_predictions.png'):
    """Visualize predictions with confidence bars."""
    num_predictions = len(predictions_data)
    
    fig = plt.figure(figsize=(16, 4 * num_predictions))
    
    for idx, pred in enumerate(predictions_data, 1):
        # Image subplot
        ax1 = plt.subplot(num_predictions, 2, 2*idx - 1)
        image = pred['image'].reshape(28, 28)
        ax1.imshow(image, cmap='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Title with actual and predicted
        color = 'green' if pred['correct'] else 'red'
        title = f"Actual: {pred['actual']} | Predicted: {pred['predicted']}\nConfidence: {pred['confidence']:.2%}"
        ax1.set_title(title, fontsize=12, fontweight='bold', color=color)
        
        # Probability bar chart
        ax2 = plt.subplot(num_predictions, 2, 2*idx)
        probabilities = pred['probabilities']
        colors = ['green' if i == pred['predicted'] else 'lightblue' for i in range(10)]
        ax2.bar(range(10), probabilities, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Digit', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax2.set_title('Prediction Probabilities', fontsize=11, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add probability values on bars
        for digit, prob in enumerate(probabilities):
            if prob > 0.05:  # Only show labels for significant probabilities
                ax2.text(digit, prob + 0.02, f'{prob:.2%}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved detailed predictions visualization to {output_path}")
    plt.close()


def print_statistics(predictions_data):
    """Print prediction statistics."""
    correct_count = sum(1 for p in predictions_data if p['correct'])
    total_count = len(predictions_data)
    accuracy = correct_count / total_count
    
    print("\n" + "="*60)
    print("PREDICTION STATISTICS")
    print("="*60)
    print(f"Total predictions: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {total_count - correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print("="*60 + "\n")


def main():
    """Main prediction pipeline."""
    print("\n" + "="*60)
    print("MNIST CNN - Prediction Script")
    print("="*60 + "\n")
    
    # Load model
    model = load_model()
    
    # Load test data
    x_test, y_test = load_test_data()
    
    # Make predictions
    predictions_data = make_predictions(model, x_test, y_test, num_predictions=5)
    
    # Visualize predictions
    os.makedirs('results', exist_ok=True)
    visualize_predictions(predictions_data)
    
    # Print statistics
    print_statistics(predictions_data)
    print("✓ Predictions complete!")
    print()


if __name__ == '__main__':
    main()
