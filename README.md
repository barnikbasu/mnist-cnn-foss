# Handwritten Digit Recognition using Convolutional Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)

## Overview

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits from the MNIST dataset. The model achieves **~98-99% accuracy** on the test set and demonstrates best practices for open-source machine learning projects.

## Problem Statement

The MNIST dataset contains 70,000 images of handwritten digits (0-9), each 28×28 pixels. The goal is to build and train a CNN that accurately recognizes and classifies these digits.

## Dataset

- **Source**: TensorFlow/Keras MNIST dataset (automatically downloaded)
- **Training samples**: 60,000
- **Test samples**: 10,000
- **Image dimensions**: 28×28 pixels (grayscale)
- **Classes**: 10 (digits 0-9)
- **Preprocessing**: Normalized to [0, 1] range

## CNN Architecture

```
Input: 28×28×1
    ↓
Conv2D (32 filters, 3×3 kernel) + ReLU
    ↓
MaxPooling (2×2)
    ↓
Conv2D (64 filters, 3×3 kernel) + ReLU
    ↓
MaxPooling (2×2)
    ↓
Flatten
    ↓
Dense (128 units) + ReLU
    ↓
Dropout (0.5)
    ↓
Dense (10 units) + Softmax
    ↓
Output: Class probabilities for digits 0-9
```

### Why This Architecture?

- **Convolutional layers**: Extract spatial features from images
- **MaxPooling**: Reduce dimensionality and retain important features
- **Dropout**: Prevent overfitting during training
- **Dense layers**: Learn high-level patterns and make predictions

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Metrics | Accuracy |
| Epochs | 10 |
| Batch Size | 128 |
| Validation Split | 10% |
| Learning Rate | 0.001 (default) |

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/barnikbasu/mnist-cnn-foss.git
   cd mnist-cnn-foss
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

```bash
python src/train.py
```

This script will:
- Download the MNIST dataset
- Normalize pixel values to [0, 1]
- Build and compile the CNN
- Train for 10 epochs with 10% validation split
- Evaluate on the test set
- Save the trained model to `models/mnist_cnn.keras`
- Generate accuracy and loss plots in `results/`
- Generate sample predictions in `results/predictions.png`
- Save metrics to `results/metrics.txt`

**Expected output**:
```
FINAL RESULTS
============================================================
Test Accuracy: 0.9890 (98.90%)
Test Loss:     0.0382
```

### Making Predictions

```bash
python src/predict.py
```

This script will:
- Load the trained model from `models/mnist_cnn.keras`
- Select 5 random test images
- Display the actual and predicted digits
- Show the model's confidence for each prediction

## Results

### Evaluation Metrics

After training, you'll find:

- `results/metrics.txt` - Test accuracy and loss
- `results/accuracy.png` - Training vs validation accuracy over epochs
- `results/loss.png` - Training vs validation loss over epochs
- `results/predictions.png` - Sample test predictions with confidence scores

### Sample Predictions

The model is evaluated on real test images. Sample predictions show:
- Actual digit labels
- Predicted digit labels
- Prediction confidence (softmax probabilities)

## Project Structure

```
mnist-cnn-foss/
├── README.md                 # Project documentation
├── SUBMISSION.md             # FOSS club submission details
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── src/
│   ├── train.py            # Training script
│   └── predict.py          # Prediction script
├── models/
│   └── mnist_cnn.keras     # Trained model (after running train.py)
└── results/
    ├── metrics.txt         # Test metrics (after training)
    ├── accuracy.png        # Accuracy plot (after training)
    ├── loss.png           # Loss plot (after training)
    └── predictions.png    # Sample predictions (after training)
```

## Challenges & Solutions

### Challenge 1: Understanding Image Dimensions After Operations
**Problem**: Tracking how image dimensions change through convolution and pooling layers.

**Solution**: 
- Conv2D(3×3 kernel, same padding) preserves dimensions
- MaxPooling(2×2) reduces dimensions by 2×2
- After first Conv2D + MaxPooling: 28×28 → 14×14
- After second Conv2D + MaxPooling: 14×14 → 7×7

### Challenge 2: Preventing Overfitting
**Problem**: Training accuracy much higher than validation accuracy.

**Solution**:
- Use Dropout(0.5) after the first Dense layer
- Monitor validation loss during training
- Use 10% validation split to detect overfitting early
- Regularization through proper architecture design

### Challenge 3: Normalizing Input Pixels
**Problem**: Raw pixel values (0-255) can slow training and cause numerical instability.

**Solution**:
```python
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
```
Normalize to [0, 1] for better gradient flow.

### Challenge 4: Choosing Appropriate Architecture
**Problem**: Too simple → poor accuracy; Too complex → overfitting and slow training.

**Solution**:
- Start with 2 Conv blocks (proven effective for MNIST)
- Moderate filter sizes (32, 64)
- Single Dense hidden layer with dropout
- This balances accuracy and computational efficiency

### Challenge 5: Reproducibility
**Problem**: Different results on different runs due to random initialization.

**Solution**:
- Set random seeds in TensorFlow and NumPy
- Use fixed training/test split
- Document all hyperparameters
- Include requirements.txt with pinned versions

## How to Run

### Local Machine

```bash
# 1. Clone and setup
git clone https://github.com/barnikbasu/mnist-cnn-foss.git
cd mnist-cnn-foss
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python src/train.py

# 4. Make predictions
python src/predict.py

# 5. Check results
ls results/
```

### Google Colab

```python
# 1. Upload mnist-cnn-foss.zip to Colab
!unzip mnist-cnn-foss.zip

# 2. Install dependencies
!pip install -r mnist-cnn-foss/requirements.txt

# 3. Train
!cd mnist-cnn-foss && python src/train.py

# 4. Predict
!cd mnist-cnn-foss && python src/predict.py

# 5. Download results
from google.colab import files
files.download('mnist-cnn-foss/results/metrics.txt')
files.download('mnist-cnn-foss/results/accuracy.png')
files.download('mnist-cnn-foss/results/loss.png')
files.download('mnist-cnn-foss/results/predictions.png')
```

## Future Improvements

- [ ] Add confusion matrix visualization
- [ ] Implement data augmentation for better generalization
- [ ] Add batch normalization layers
- [ ] Experiment with different optimizers (SGD, RMSprop)
- [ ] Implement learning rate scheduling
- [ ] Add support for custom digit images (webcam or upload)
- [ ] Create a simple web interface for predictions
- [ ] Add model quantization for mobile deployment
- [ ] Implement cross-validation for more robust evaluation
- [ ] Add comprehensive unit tests

## Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [Convolutional Neural Networks (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [Adam Optimizer](https://arxiv.org/abs/1412.6980)
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)

## Author

**Barnik Basu**
- GitHub: [@barnikbasu](https://github.com/barnikbasu)

---

**Last Updated**: October 2026
**Status**: Complete and tested ✓
