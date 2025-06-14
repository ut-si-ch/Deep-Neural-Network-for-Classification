
# Deep Neural Network for Classification (Fashion MNIST Dataset)

## 🧾 Overview

This project builds and evaluates a **Deep Neural Network (DNN)** using **Keras** to classify images from the **Fashion MNIST** dataset. The DNN model learns to distinguish among various fashion items such as shirts, shoes, and bags based on pixel-level image data.

---

## 🧠 Business Understanding

Image classification is foundational for many real-world applications in:
- E-commerce (product tagging and recommendations)
- Healthcare (diagnostic imaging)
- Manufacturing (defect detection)

This project simulates how a deep learning pipeline can be applied to a structured visual dataset to automate classification tasks efficiently.

---

## 📊 Data Understanding

Dataset used: **Fashion MNIST** (from Keras datasets)

- 60,000 training images + 10,000 test images
- 28x28 grayscale images
- 10 fashion categories:
  - T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

### Preprocessing:
- Normalization (pixel values scaled to 0-1)
- Reshaping images for dense layers
- One-hot encoding of labels

---

## 🤖 Modelling & Evaluation

### Deep Neural Network Architecture:
- Input Layer: Flattened 784 features
- Hidden Layers: Dense layers with ReLU activation
- Output Layer: 10 units (Softmax for multiclass classification)

### Training:
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Evaluation Metrics: Accuracy

### Evaluation:
- Training vs validation loss and accuracy visualization
- Confusion Matrix
- Classification Report with precision, recall, F1-score

---

## 📌 Conclusion

- The model effectively classifies fashion items with high accuracy
- Demonstrates robustness in handling image data with a simple dense neural network
- Strong foundational project for advancing to CNNs and more complex architectures

---

## ✅ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- Scikit-learn (for evaluation)

---
