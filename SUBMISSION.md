# FOSS Club Submission: Handwritten Digit Recognition CNN

## Project Information

* **Project Name:** Handwritten Digit Recognition using Convolutional Neural Networks
* **Repository:** https://github.com/barnikbasu/mnist-cnn-foss
* **Author:** Barnik Basu
* **License:** MIT
* **Language:** Python
* **Framework:** TensorFlow/Keras
* **Dataset:** MNIST

## Problem Statement

MNIST is a standard handwritten-digit classification dataset containing 70,000 grayscale images of digits from 0 to 9.

The objective of this project is to build a Convolutional Neural Network capable of learning visual features from handwritten digits and accurately classifying previously unseen test images.

## Approach

I implemented a lightweight two-block CNN using TensorFlow/Keras.

The design focuses on:

1. **Feature extraction** using convolutional layers
2. **Dimensionality reduction** using max-pooling
3. **Classification** using dense layers
4. **Regularization** using dropout
5. **Optimization** using Adam
6. **Evaluation** using a held-out MNIST test set

The implementation is intentionally simple so that the architecture and training process remain easy to understand and reproduce.

## Technical Details

### Architecture

```text
Input: 28×28×1
        ↓
Conv2D: 32 filters, 3×3, valid padding + ReLU
        ↓
MaxPooling2D: 2×2
        ↓
13×13×32
        ↓
Conv2D: 64 filters, 3×3, valid padding + ReLU
        ↓
MaxPooling2D: 2×2
        ↓
5×5×64
        ↓
Flatten
        ↓
Dense: 128 units + ReLU
        ↓
Dropout: 0.5
        ↓
Dense: 10 units + Softmax
        ↓
Output: digits 0–9
```

### Dimension Calculation

The model uses `valid` padding for the convolutional layers.

```text
Input
28×28×1

First Conv2D (3×3)
26×26×32

First MaxPooling (2×2)
13×13×32

Second Conv2D (3×3)
11×11×64

Second MaxPooling (2×2)
5×5×64

Flatten
5 × 5 × 64 = 1600 features
```

### Why This Architecture?

* **Two convolutional layers** provide hierarchical feature extraction.
* **32 → 64 filters** increase feature capacity in the deeper layer.
* **Max-pooling** reduces spatial dimensions and computational cost.
* **Dense(128)** performs higher-level classification.
* **Dropout(0.5)** provides regularization.
* **Softmax** converts the final outputs into class probabilities.

The architecture provides a practical balance between model simplicity, computational cost, and classification performance for MNIST.

## Training Configuration

| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Optimizer         | Adam                            |
| Loss              | Sparse Categorical Crossentropy |
| Metric            | Accuracy                        |
| Epochs            | 10                              |
| Batch Size        | 128                             |
| Validation Split  | 10%                             |
| Random Seed       | 42                              |
| Number of Classes | 10                              |

## Results

A recorded training run achieved:

```text
Test Accuracy: 0.9925 (99.25%)
Test Loss:     0.0255
```

These values represent the recorded experiment and are not intended as a guaranteed result for every environment.

Small variations can occur because of differences in TensorFlow versions, hardware, and numerical behavior.

## Generated Artifacts

The repository contains:

```text
results/
├── accuracy.png
├── loss.png
├── predictions.png
└── metrics.txt
```

### `accuracy.png`

Shows training and validation accuracy over the 10 training epochs.

### `loss.png`

Shows training and validation loss over the 10 training epochs.

### `predictions.png`

Shows sample MNIST test images with:

* Actual label
* Predicted label
* Prediction confidence

### `metrics.txt`

Contains the recorded test accuracy, test loss, and training configuration.

## Google Colab Experiment

The complete experiment is also available through a Google Colab notebook.

**Colab:**
`PASTE_YOUR_COLAB_LINK_HERE`

The notebook should use the current GitHub repository as its source so that the code, experiment, and generated results remain aligned.

## Challenges Faced

### 1. Understanding Dimension Transformations

One of the main challenges was understanding how convolution and pooling change image dimensions.

Because the model uses `valid` padding, each 3×3 convolution reduces the spatial dimensions.

The final feature-map size is:

```text
28×28
→ 26×26
→ 13×13
→ 11×11
→ 5×5
```

This results in:

```text
5 × 5 × 64 = 1600
```

features before the dense layer.

### 2. Preventing Overfitting

The model can achieve very high training accuracy, so monitoring validation performance is important.

I used:

* 10% validation split
* Dropout(0.5)
* A relatively compact architecture
* Separate test-set evaluation

### 3. Data Normalization

MNIST pixels originally range from 0 to 255.

They are converted to floating-point values between 0 and 1 before training:

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```

This provides a more suitable input range for neural-network optimization.

### 4. Choosing the Architecture

The challenge was to avoid making the model unnecessarily complicated while still obtaining strong performance.

I chose a two-block CNN with 32 and 64 filters because it provides enough capacity for MNIST while remaining lightweight and easy to understand.

### 5. Reproducibility

Training and prediction use a fixed random seed of 42.

The main hyperparameters are documented, and the repository provides a complete training script that automatically downloads MNIST and generates the experiment artifacts.

## FOSS Project Quality

### Open-Source Structure

* [x] MIT License
* [x] README documentation
* [x] Clear source-code organization
* [x] Dependency declaration
* [x] `.gitignore`
* [x] Documented project structure
* [x] Results and experiment artifacts

### Reproducibility

* [x] Automatic MNIST dataset download
* [x] No hardcoded local dataset paths
* [x] Fixed random seed
* [x] Documented hyperparameters
* [x] Complete training script
* [x] Google Colab experiment
* [x] Generated evaluation artifacts

### Best Practices

* [x] Source code separated from generated results
* [x] MIT open-source license
* [x] Comments and docstrings for important code
* [x] Honest reporting of experimental results
* [x] Documented challenges
* [x] Documented future improvements

## How to Run

### Clone

```bash
git clone https://github.com/barnikbasu/mnist-cnn-foss.git
cd mnist-cnn-foss
```

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python src/train.py
```

### Predict

```bash
python src/predict.py
```

## Future Improvements

Potential future improvements include:

1. Confusion matrix visualization
2. Data augmentation
3. Batch normalization
4. Learning-rate scheduling
5. Custom handwritten-image input
6. Web-based prediction interface
7. TensorFlow Lite conversion
8. Automated testing
9. Continuous integration

These are intentionally treated as future extensions rather than unnecessary complexity in the current implementation.

## Why This Project Demonstrates FOSS Competency

This project demonstrates:

1. **Machine-learning fundamentals** — preprocessing, CNN architecture, training, validation, and evaluation.
2. **Software engineering** — modular Python scripts, dependency management, documentation, and project structure.
3. **Reproducibility** — fixed seed, documented configuration, and an executable training pipeline.
4. **Open-source practices** — MIT licensing, documentation, transparent results, and future contribution paths.
5. **Learning ability** — the project documents the practical challenges encountered while implementing and understanding the CNN.

## Contact

* **GitHub:** https://github.com/barnikbasu
* **Repository:** https://github.com/barnikbasu/mnist-cnn-foss

---

**Submission Date:** October 2026
**Status:** Ready for evaluation after the documented experiment is reproduced.
