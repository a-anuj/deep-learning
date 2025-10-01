# 🧠 Deep Learning Projects: ANN & CNN

This repo contains two projects demonstrating the use of **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)**.  

- **Project 1: ANN on Churn Modelling Dataset**
- **Project 2: CNN on CIFAR-10 Dataset**

---

## 📌 Project 1: Artificial Neural Network (ANN) - Churn Modelling

### 🔹 Overview
The **Churn Modelling dataset** (Bank Customers dataset) is used to predict whether a customer will leave the bank (churn) or not, based on their details such as geography, gender, credit score, balance, number of products, etc.

### 🛠️ Steps
1. Preprocessed the dataset (encoding categorical data, feature scaling).
2. Built a feed-forward ANN using **TensorFlow/Keras**.
3. Used `Dense` layers with ReLU and Sigmoid activations.
4. Optimized using **Adam optimizer** and **binary cross-entropy** loss.
5. Evaluated using accuracy, confusion matrix, and classification report.

### ⚙️ Model Architecture
- Input layer (features from dataset)
- Hidden layers (ReLU activation)
- Output layer (Sigmoid for binary classification)



## 📌 Project 2: Convolutional Neural Network (CNN) - CIFAR-10

### 🔹 Overview
The **CIFAR-10 dataset** consists of **60,000 32x32 color images** across **10 classes** (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).  
Goal: Build a CNN that classifies images into these categories.

### 🛠️ Steps
1. Loaded and normalized CIFAR-10 dataset.
2. Built a CNN model using **Conv2D, MaxPooling2D, Flatten, Dense, Dropout** layers.
3. Used `ReLU` activations in hidden layers and `Softmax` in the output layer.
4. Optimized with **Adam optimizer** and **categorical cross-entropy** loss.
5. Evaluated with accuracy and loss graphs.

### ⚙️ Model Architecture
- Conv2D + ReLU + MaxPooling
- Conv2D + ReLU + MaxPooling
- Flatten layer
- Dense layers with ReLU
- Output Dense layer with Softmax (10 classes)

### 📊 Results
- Training Accuracy: ~85-90%
- Test Accuracy: ~80-85%  
- The model can classify CIFAR-10 images with good accuracy.


## ⚡ Tech Stack
- **Python 3**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **Scikit-learn**

