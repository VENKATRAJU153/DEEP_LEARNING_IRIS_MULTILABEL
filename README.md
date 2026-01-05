# DEEP_LEARNING_IRIS_MULTILABEL
Below is a **DETAILED, PROFESSIONAL GitHub README.md explanation** for your **Iris Flower Classification using ANN** project.
You can **directly copy–paste** this into your GitHub `README.md`.

---

# 🌸 Iris Flower Classification using Artificial Neural Network (ANN)

## 📌 Project Overview

This project implements an **Artificial Neural Network (ANN)** using **TensorFlow & Keras** to classify Iris flowers into three species:

* 🌼 **Iris Setosa**
* 🌺 **Iris Versicolor**
* 🌸 **Iris Virginica**

The model learns from four numerical features of the flowers and predicts the species with high accuracy.

---

## 🎯 Problem Statement

Given flower measurements, predict the **species of Iris flower** using a deep learning model.

This is a **multi-class classification problem** with three output classes.

---

## 📂 Dataset Description

The project uses the **Iris Dataset**, which contains **150 samples** and **5 columns**:

| Column Name   | Description                   |
| ------------- | ----------------------------- |
| SepalLengthCm | Sepal length (cm)             |
| SepalWidthCm  | Sepal width (cm)              |
| PetalLengthCm | Petal length (cm)             |
| PetalWidthCm  | Petal width (cm)              |
| Species       | Target variable (flower type) |

🔹 The `Id` column is dropped as it has no predictive value.

---

## 🔄 Data Preprocessing

### 1️⃣ Dropping Unnecessary Columns

```python
df = df.drop(['Id'], axis=1)
```

### 2️⃣ Encoding Target Variable

The categorical species labels are converted into numerical values:

```python
'Iris-setosa'     → 0  
'Iris-versicolor' → 1  
'Iris-virginica'  → 2
```

### 3️⃣ Feature & Target Split

```python
X = df.iloc[:, :-1]   # Independent variables
y = df.iloc[:, -1]    # Dependent variable
```

---

## ✂️ Data Splitting

The dataset is manually split into:

* **Training Data:** 130 samples
* **Validation Data:** 15 samples
* **Test Data:** 5 samples

```python
Train → Used for learning  
Validation → Used for tuning  
Test → Used for final prediction
```

---

## 🧠 Model Architecture (ANN)

The ANN is built using **Sequential API** with multiple dense layers.

### 🔹 Architecture Summary

| Layer          | Neurons | Activation |
| -------------- | ------- | ---------- |
| Input Layer    | 4       | —          |
| Hidden Layer 1 | 128     | ReLU       |
| Hidden Layer 2 | 64      | ReLU       |
| Hidden Layer 3 | 32      | ReLU       |
| Hidden Layer 4 | 8       | ReLU       |
| Hidden Layer 5 | 4       | ReLU       |
| Hidden Layer 6 | 2       | ReLU       |
| Output Layer   | 3       | Softmax    |

✔ **ReLU** activation improves non-linearity
✔ **Softmax** outputs probability distribution for 3 classes
✔ **He Uniform** initialization improves convergence

---

## ⚙️ Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['Accuracy']
)
```

* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Metric:** Accuracy

---

## 🔁 One-Hot Encoding

The target labels are converted into categorical format for training:

```python
to_categorical(y)
```

---

## 🚀 Model Training

```python
epochs = 50  
batch_size = 20
```

The model is trained using:

* Training data for learning
* Validation data for performance monitoring

---

## 📊 Performance Visualization

Two plots are generated:

### 📈 Training Performance

* Training Accuracy
* Training Loss

### 📉 Validation Performance

* Validation Accuracy
* Validation Loss

These graphs help detect:

* Overfitting
* Underfitting
* Model convergence behavior

---

## 🔮 Predictions

### ✅ Single Sample Prediction

```python
[6.7, 3.0, 5.2, 2.3] → Virginica
```

### ✅ Test Data Prediction

The model predicts species labels using:

```python
np.argmax(prediction)
```

Output labels:

* `0 → Setosa`
* `1 → Versicolor`
* `2 → Virginica`

---

## 🧪 Results

* ✔ High accuracy on training and validation data
* ✔ Correct classification of unseen test samples
* ✔ Stable loss reduction over epochs

---

## 🛠️ Technologies Used

* Python 🐍
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras

---

## 📌 Key Learnings

* ANN architecture design
* Multi-class classification using softmax
* One-hot encoding
* Model evaluation using accuracy and loss
* Visualization of learning curves

---

## 🔮 Future Enhancements

* Use `train_test_split` with shuffling
* Reduce model complexity
* Add confusion matrix & classification report
* Hyperparameter tuning
* Deploy model using Flask or Streamlit



Just tell me 👍
