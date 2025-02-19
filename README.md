# Titanic-ML: Logistic Regression from Scratch  

### Manually Implemented Optimization Algorithms (No ML Libraries Used!)

This project builds a **Titanic survival prediction model** using **logistic regression** trained **from scratch**.  
Unlike traditional ML workflows that rely on `scikit-learn` or `TensorFlow`, this project **implements optimization manually** using:  

**Batch Gradient Descent (from scratch)**  
**Newton’s Method for Optimization (manual implementation)**  
**Feature Normalization & One-hot Encoding**  
**NumPy-only Computation (No External ML Libraries Used!)**  

This makes the project an excellent **learning resource** for those who want to understand ML algorithms **at a fundamental level**.

---

## Overview  

This repository contains a **fully implemented logistic regression model** for predicting survival on the Titanic dataset from Kaggle.  

### Key Features  

- **Handcrafted Optimization Algorithms** – Implements **batch gradient descent** and **Newton’s method** without ML libraries.  
- **Feature Engineering** – Converts categorical data using **one-hot encoding**, applies **feature scaling**, and extracts **useful attributes**.  
- **Cross-Validation** – Evaluates the model with **k-fold cross-validation** to find the optimal regularization parameter (`lambda`).  
- **Custom Accuracy Evaluation** – Computes accuracy manually for both **train and test sets**.  

---

##  Model Performance  

**Train Accuracy:** 80 - 85 %  
**Test Accuracy:**  76 - 81 %  

The model achieves **competitive results** using purely **handcrafted** optimization techniques.

---

## 🛠 Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/MansMuzafarov/Titanic-ML.git
   cd Titanic-ML

2. Install dependencies:

pip install -r requirements.txt

3. Running the model:

python main.py


## Project structure:

├── Data/ # Contains train and test datasets 
├── data_preprocessing.py  # Data cleaning & feature engineering  
├── features_designs.py    # Feature transformation functions  
├── model.py               # Logistic regression & classification functions  
├── optimization.py        # Gradient Descent & Newton’s Method  
├── calculate_accuracy.py  # Train & Test accuracy evaluation  
├── train_and_test.py      # Model training & evaluation pipeline  
├── main.py                # Main script to run the model  
└── README.md              # Project documentation  






# Data Preprocessing:

Cleans Titanic dataset 
Encodes categorical variables 
Normalizes numerical features

# Model Training & Optimization:

Uses batch gradient descent & Newton's method
Runs k-fold cross-validation to find the best lambda (regularization parameter)

# Evaluation:

Computes train/test accuracy
Makes final predictions

# License
This project is licensed under the MIT License.

# Contributing
Feel free to fork, improve, and submit pull requests! Suggestions and optimizations are welcome.
