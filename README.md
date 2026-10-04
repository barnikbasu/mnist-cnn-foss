# Handwritten Digit Recognition using Convolutional Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)

## Overview

This project implements a Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset.

The project focuses on a simple, understandable, and reproducible machine learning workflow:

* Data loading and preprocessing
* CNN architecture design
* Model training and validation
* Test-set evaluation
* Prediction visualization
* Reproducible experiment configuration
* Open-source documentation and licensing

A recorded training run achieved **99.25% test accuracy** with a test loss of **0.0255**.

> Note: Results can vary slightly across environments because of differences in TensorFlow versions, hardware, and numerical behavior.

## Problem Statement

The MNIST dataset contains 70,000 grayscale images of handwritten digits from 0 to 9.

Each image is 28×28 pixels.

The objective is to build a CNN that can learn visual features from these images and correctly classify each image into one of the ten digit classes.

## Dataset

The project uses the MNIST dataset provided through TensorFlow/Keras.

| Property                        | Value               |
| ------------------------------- | ------------------- |
| Total images                    | 70,000              |
| Training images                 | 60,000              |
| Test images                     | 10,000              |
| Image size                      | 28×28 pixels        |
| Channels                        | 1 grayscale channel |
| Classes                         | 10 (digits 0–9)     |
| Pixel range after preprocessing | 0–1                 |

The dataset is downloaded automatically when the training script is executed.

## Preprocessing

The original MNIST pixel values are integers in the range:

```text
0 to 255
```

They are normalized to:

```text
0 to 1
```

using:

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```

A channel dimension is then added:

```text
28×28 → 28×28×1
```

This gives the CNN an explicit grayscale-channel dimension.

## CNN Architecture

The model uses two convolutional blocks followed by a fully connected classifier.

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
Output: probabilities for digits 0–9
```

### Why these dimensions?

The convolution layers use Keras' `valid` padding by default.

For a 3×3 convolution:

```text
28×28 → 26×26
```

After 2×2 max pooling:

```text
26×26 → 13×13
```

The second 3×3 convolution gives:

```text
13×13 → 11×11
```

The second 2×2 max-pooling operation gives:

```text
11×11 → 5×5
```

Therefore the final convolutional output is:

```text
5×5×64
```

which becomes:

```text
1600 features
```

after flattening.

## Why This Architecture?

The architecture was chosen as a simple and effective baseline for MNIST.

* **Convolutional layers** learn spatial features such as edges and shapes.
* **MaxPooling** reduces spatial dimensions while retaining important features.
* **ReLU** provides nonlinear activation.
* **Flatten** converts feature maps into a vector for classification.
* **Dense layer** learns higher-level combinations of extracted features.
* **Dropout** reduces over-reliance on individual neurons.
* **Softmax** produces probabilities for the ten digit classes.

The model is intentionally lightweight rather than over-engineered for a relatively simple benchmark dataset.

## Training Configuration

| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Optimizer         | Adam                            |
| Loss              | Sparse Categorical Crossentropy |
| Metric            | Accuracy                        |
| Epochs            | 10                              |
| Batch size        | 128                             |
| Validation split  | 10%                             |
| Random seed       | 42                              |
| Input size        | 28×28×1                         |
| Number of classes | 10                              |

The Adam optimizer uses its standard Keras default learning rate.

## Results

A recorded training run produced:

```text
Test Accuracy: 0.9925 (99.25%)
Test Loss:     0.0255
```

### Generated Results

The repository contains:

* `results/accuracy.png` — training and validation accuracy
* `results/loss.png` — training and validation loss
* `results/predictions.png` — sample predictions
* `results/metrics.txt` — recorded evaluation metrics

The exact result can vary slightly when training is repeated in a different environment.

## Google Colab Experiment

The complete training experiment is also available as a Google Colab notebook.

**Colab Notebook:** https://colab.research.google.com/drive/13zGFxU52UIeRjYQJ5UWtOd6Dcb_n9-X5

The notebook uses the GitHub repository as its source so that the experiment and repository remain aligned.

## Project Structure

```text
mnist-cnn-foss/
│
├── README.md
├── SUBMISSION.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── MNIST_CNN_FOSS_Club_IIIT_Kalyani.ipynb
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
│
└── results/
    ├── accuracy.png
    ├── loss.png
    ├── metrics.txt
    └── predictions.png
```

The trained model is generated locally as:

```text
models/mnist_cnn.keras
```

It is intentionally not required to be committed to the repository because the training script can recreate it.

## Installation

### Requirements

* Python 3.8 or newer
* pip
* Internet connection for the first MNIST dataset download

### Setup

Clone the repository:

```bash
git clone https://github.com/barnikbasu/mnist-cnn-foss.git
cd mnist-cnn-foss
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Train the Model

Run:

```bash
python src/train.py
```

The script will:

1. Download the MNIST dataset.
2. Normalize the input images.
3. Build the CNN.
4. Train for 10 epochs.
5. Use 10% of the training data for validation.
6. Evaluate the model on the test set.
7. Generate accuracy and loss plots.
8. Generate sample predictions.
9. Save the trained model.
10. Save evaluation metrics.

Generated files:

```text
models/mnist_cnn.keras
results/accuracy.png
results/loss.png
results/predictions.png
results/metrics.txt
```

### Make Predictions

After training:

```bash
python src/predict.py
```

The prediction script:

* Loads the trained model.
* Loads the MNIST test set.
* Selects five reproducible test samples.
* Displays actual and predicted labels.
* Displays confidence scores.
* Generates a detailed prediction visualization.

The standalone prediction visualization is saved as:

```text
results/sample_predictions.png
```

## Google Colab

The repository can also be run directly in Google Colab.

Clone the repository:

```python
!git clone https://github.com/barnikbasu/mnist-cnn-foss.git
```

Install dependencies:

```python
!pip install -q -r mnist-cnn-foss/requirements.txt
```

Train:

```python
!cd mnist-cnn-foss && python src/train.py
```

Run predictions:

```python
!cd mnist-cnn-foss && python src/predict.py
```

View the metrics:

```python
!cat mnist-cnn-foss/results/metrics.txt
```

## Challenges and Solutions

### 1. Understanding Dimension Changes

A challenge was understanding how convolution and pooling affect image dimensions.

The model uses `valid` padding, so each 3×3 convolution reduces the spatial dimensions.

The main transformations are:

```text
28×28
  ↓ Conv 3×3
26×26
  ↓ Pool 2×2
13×13
  ↓ Conv 3×3
11×11
  ↓ Pool 2×2
5×5
```

Understanding these transformations helped determine the size of the flattened feature vector.

### 2. Preventing Overfitting

A CNN can achieve very high training accuracy while performing worse on unseen data.

To improve generalization, the model uses:

* A 10% validation split
* Dropout with a rate of 0.5
* A relatively small architecture
* Separate evaluation on the held-out test set

### 3. Input Normalization

Raw pixel values range from 0 to 255.

Normalizing them to 0–1 provides a more suitable numerical range for neural network training.

### 4. Architecture Selection

The architecture was designed as a lightweight baseline with:

* Two convolutional layers
* 32 and 64 filters
* Two max-pooling layers
* One hidden dense layer
* Dropout regularization

This provides a good balance between simplicity, training cost, and MNIST classification performance.

### 5. Reproducibility

The project uses a fixed random seed:

```text
42
```

for NumPy and TensorFlow.

The prediction sample selection is also seeded.

The repository documents the main hyperparameters and provides a complete training script so that the experiment can be reproduced.

## Future Improvements

Possible extensions include:

* [ ] Add a confusion matrix
* [ ] Experiment with data augmentation
* [ ] Add batch normalization
* [ ] Experiment with different optimizers
* [ ] Add learning-rate scheduling
* [ ] Support custom uploaded handwritten digits
* [ ] Build a simple web interface
* [ ] Convert the model to TensorFlow Lite
* [ ] Add automated tests
* [ ] Add CI checks for code quality

These are intentionally left as future improvements rather than unnecessary additions to the current baseline.

## FOSS Practices

This repository follows several open-source practices:

* MIT License
* Clear README documentation
* Reproducible training script
* Dependency declaration
* Clean project structure
* Source code separated from generated results
* Documented limitations and future improvements
* No proprietary dataset or closed-source dependency required for the core experiment

## Contributing

Contributions are welcome.

A typical workflow is:

```bash
git checkout -b feature/improvement
```

Make your changes, test them, and commit:

```bash
git add .
git commit -m "Describe the change"
```

Push the branch and open a pull request.

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for the complete license text.

## References

* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
* [TensorFlow Documentation](https://www.tensorflow.org/)
* [Keras Documentation](https://keras.io/)
* [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)
* [Adam Optimizer](https://arxiv.org/abs/1412.6980)
* [Dropout](https://jmlr.org/papers/v15/srivastava14a.html)

## Author

**Barnik Basu**

GitHub: https://github.com/barnikbasu

Repository: https://github.com/barnikbasu/mnist-cnn-foss

---

**Status:** Ready for evaluation after the documented experiment is reproduced.
**Last Updated:** October 2026
