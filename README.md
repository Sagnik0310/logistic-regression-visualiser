# 📈 Logistic Regression Visualizer

An interactive **Logistic Regression Visualizer** built with **Python**, **Streamlit**, **Scikit-learn**, and **Matplotlib**. This application enables users to explore how different Logistic Regression hyperparameters affect the decision boundary, classification performance, and model behavior on both binary and multiclass datasets.

---

## 🚀 Features

- 📊 Interactive visualization of Binary and Multi-class datasets
- 🎯 Real-time decision boundary visualization
- ⚙️ Adjustable Logistic Regression hyperparameters
  - Regularization (`L1`, `L2`, `Elastic Net`, `None`)
  - Regularization Strength (`C`)
  - Solver Selection
  - Maximum Iterations
  - L1 Ratio (Elastic Net)
- 📈 Accuracy calculation on the test dataset
- 🌈 Beautiful Streamlit interface with dynamic plots

---

## 📂 Project Structure

```
logistic-regression-visualiser/

├── app.py
└── README.md
```

---

# 🧠 Logistic Regression

Logistic Regression is a supervised learning algorithm used for **classification** problems.

Unlike Linear Regression, Logistic Regression predicts the **probability** of belonging to a class.

The hypothesis function is

\[
h_\theta(x)=\frac{1}{1+e^{-z}}
\]

where

\[
z=\theta^Tx+b
\]

The sigmoid function converts any real-valued number into a probability between **0 and 1**.

---

# 📉 Sigmoid Function

The sigmoid function is defined as

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

where

- \(z\) = Linear combination of features
- Output lies in the interval

\[
0 \le \sigma(z) \le 1
\]

Prediction rule:

\[
\hat y=
\begin{cases}
1 & \text{if } \sigma(z)\ge0.5\\
0 & \text{otherwise}
\end{cases}
\]

---

# 📌 Cost Function

Since Logistic Regression performs classification, Mean Squared Error is not suitable.

Instead, it minimizes the **Log Loss (Cross Entropy Loss)**

\[
J(\theta)=
-\frac1m
\sum_{i=1}^{m}
\left[
y_i\log(h_\theta(x_i))
+
(1-y_i)
\log(1-h_\theta(x_i))
\right]
\]

The optimizer updates the parameters until this cost is minimized.

---

# 📌 Decision Boundary

The decision boundary separates different classes.

For Binary Classification,

\[
\theta^Tx+b=0
\]

Points satisfying

\[
\theta^Tx+b>0
\]

belong to one class while

\[
\theta^Tx+b<0
\]

belong to the other.

The application visualizes this boundary dynamically after training.

---

# 📚 Regularization

Regularization prevents the model from overfitting by penalizing large parameter values.

---

## 🔹 L1 Regularization (Lasso)

Penalty:

\[
\lambda
\sum_{j=1}^{n}
|\theta_j|
\]

Characteristics

- Performs feature selection
- Can reduce coefficients exactly to zero
- Produces sparse models

---

## 🔹 L2 Regularization (Ridge)

Penalty:

\[
\lambda
\sum_{j=1}^{n}
\theta_j^2
\]

Characteristics

- Shrinks coefficients
- Keeps all features
- Reduces model variance

---

## 🔹 Elastic Net

Combination of L1 and L2

\[
J(\theta)
=
Loss
+
\lambda_1
\sum|\theta|
+
\lambda_2
\sum\theta^2
\]

The balance is controlled using

\[
l1\_ratio
\]

where

- 0 → Pure L2
- 1 → Pure L1
- 0.5 → Equal contribution of both

---

# ⚙️ Hyperparameters

| Hyperparameter | Description |
|---------------|-------------|
| Dataset | Binary or Multi-class synthetic dataset |
| Penalty | Regularization method |
| C | Inverse of regularization strength |
| Solver | Optimization algorithm |
| Max Iterations | Maximum optimization iterations |
| L1 Ratio | Balance between L1 and L2 (Elastic Net only) |

---

# 📊 Supported Solvers

| Solver | L1 | L2 | Elastic Net | None |
|---------|:--:|:--:|:-----------:|:----:|
| liblinear | ✅ | ✅ | ❌ | ❌ |
| lbfgs | ❌ | ✅ | ❌ | ✅ |
| newton-cg | ❌ | ✅ | ❌ | ✅ |
| sag | ❌ | ✅ | ❌ | ✅ |
| saga | ✅ | ✅ | ✅ | ✅ |

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/logistic-regression-visualiser.git
```

Move into the project

```bash
cd logistic-regression-visualiser
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- NumPy
- Matplotlib

---

---

# 🎯 Learning Outcomes

This project demonstrates:

- Logistic Regression fundamentals
- Sigmoid Function
- Binary Classification
- Multiclass Classification
- Decision Boundary Visualization
- Regularization Techniques
- Hyperparameter Tuning
- Streamlit Dashboard Development

---

# 📖 References

- Scikit-learn Documentation
- Streamlit Documentation
- Matplotlib Documentation

---

## ⭐ If you found this project useful, consider giving it a star!