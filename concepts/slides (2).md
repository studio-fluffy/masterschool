# Session — SLIDES: Data Science 101

---

### Slide 1 — Title

**Data Science 101: From Data to Decisions**
Understanding DS, ML, and AI — and how to actually build something

---

### Slide 2 — The Ecosystem Is Confusing

You have probably seen these terms used interchangeably:

* Data Analyst
* Data Scientist
* Machine Learning Engineer
* AI Engineer

They are not the same thing. Today we build a clear mental map.

---

### Slide 3 — The Map

From largest to smallest scope:

* **AI** — the broad goal: machines that do intelligent things
* **Data Science** — extract insights from data using stats, visualisation, and ML
* **Machine Learning** — algorithms that improve automatically from data
* **Deep Learning** — ML using multi-layer neural networks

Every ML Engineer does Data Science. Not every Data Scientist does ML.

---

### Slide 4 — Examples

Concrete examples help:

* Your phone's face unlock → Deep Learning
* Netflix recommendations → Machine Learning
* A sales bar chart → Data Science
* A rules-based spam filter → AI, but not ML

---

### Slide 5 — The ML Workflow

Every ML project follows the same loop:

1. Get data
2. Clean the data
3. Explore and visualise (EDA)
4. Engineer features
5. Split into train / validation / test sets
6. Train a model
7. Evaluate on the validation set
8. Tune and iterate

This is not linear — most projects loop back to step 2 many times.

---

### Slide 6 — Supervised Learning

The model learns from labelled examples.

* You provide inputs and the correct outputs
* The model learns to map inputs to outputs
* **Regression** — predict a number (house price, temperature)
* **Classification** — predict a category (spam / not spam, cancer type)

Example: 10,000 emails labelled as spam or not spam → train a spam detector

---

### Slide 7 — Unsupervised Learning

The model finds structure in data without labels.

* **Clustering** — group similar items together (customer segments)
* **Dimensionality reduction** — compress data while keeping structure (PCA, t-SNE)

Example: group 100,000 customers by purchase behaviour — no labels needed

---

### Slide 8 — Linear and Logistic Regression

The simplest and most interpretable models.

* **Linear regression** — predicts a continuous number
  * Good for: house prices, sales forecasting, tip prediction
  * Not good for: non-linear patterns, classification
* **Logistic regression** — predicts a binary category (0 or 1)
  * Good for: churn prediction, spam detection, credit risk
  * Not good for: complex decision boundaries

Both are fast, transparent, and a great starting point.

---

### Slide 9 — Tree-Based Models

Decision trees, random forests, and gradient boosting.

* **Decision tree** — splits data by feature values, easy to explain
  * Gets good at: quick prototypes and explainable rules
  * Problem: overfits easily

* **Random forest** — many trees averaged together, more robust

* **Gradient boosting** (XGBoost, LightGBM) — sequential trees, state of the art on tabular data
  * Wins most Kaggle competitions on structured data
  * Needs tuning, slower to train

---

### Slide 10 — Neural Networks

Layers of connected neurons that transform inputs.

* Each layer learns increasingly abstract patterns
* Shine on unstructured data: images, audio, text
* Require large amounts of data (100k+ examples to be effective)
* Are harder to interpret than tree-based models

Variants:
* **CNN** — images
* **RNN / LSTM** — sequences and time series
* **Transformer** — text (the backbone of GPT, BERT)

For tabular data, gradient boosting usually wins.

---

### Slide 11 — Your Python Toolkit

The core libraries:

* **pandas** — load, clean, and transform data
* **numpy** — numerical operations and arrays
* **matplotlib / seaborn** — visualisation
* **scikit-learn** — the standard ML library: models, metrics, pipelines
* **xgboost / lightgbm** — gradient boosting
* **tensorflow / pytorch** — deep learning
* **statsmodels / prophet** — time series

---

### Slide 12 — Regression Metrics

How to measure a model that predicts numbers:

* **RMSE** — average error in the same unit as the target; penalises large errors
* **MAE** — average absolute error; more robust to outliers
* **R²** — percentage of variance the model explains (1.0 = perfect, 0 = no better than the mean)

Example: predicting tip amounts in dollars — RMSE of $1.00 means we are off by about a dollar on average.

---

### Slide 13 — Classification Metrics

How to measure a model that predicts categories:

* **Accuracy** — percentage of correct predictions (misleading with imbalanced data)
* **Precision** — of all cases flagged positive, how many actually were?
* **Recall** — of all actual positives, how many did we catch?
* **F1 score** — balance between precision and recall
* **AUC-ROC** — how well the model ranks predictions across thresholds

For cancer detection, recall matters more than precision — missing a real case is worse than a false alarm.

---

### Slide 14 — Session Plan

What we will build together today:

1. **Linear regression** — predict restaurant tips, evaluate with RMSE and R²
2. **Feature engineering** — improve the model by creating new columns
3. **Gradient boosting** — classify cancer cases, read a full classification report
4. **Cross-validation** — confirm our model is trustworthy

---

## Live Demo: Predicting Tips and Detecting Cancer

### Demo Goal

Show students the full ML workflow from loading data to evaluating a model — twice: once with regression, once with classification.

By the end, students should understand:

* How to train a model in under 10 lines of code
* What RMSE, R², precision, and recall mean in practice
* How feature engineering improves a model without adding new data
* Why cross-validation matters

---

### Part 1: Linear Regression on the Tips Dataset

#### 1. Load and Explore the Data

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

print(tips.shape)      # 244 rows, 7 columns
print(tips.dtypes)
tips.head()
```

Explain the columns:

* `total_bill` — the restaurant bill
* `tip` — what we want to predict
* `size` — number of people at the table
* `time`, `day`, `sex`, `smoker` — context we could use as features

```python
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.title("Bill vs Tip")
plt.show()
```

Ask students: "Does this look linear? What do you notice about the spread?"

---

#### 2. Train a Linear Regression Model

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

X = tips[["total_bill", "size"]]
y = tips["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE : ${rmse:.2f}")
print(f"R²   : {r2:.2f}")
```

Expected output:
```
RMSE : $1.03
R²   : 0.47
```

Explain:

"RMSE of $1.03 — we're off by about a dollar on average. R² of 0.47 means we explain 47% of the variance. Not great. Can we do better?"

---

#### 3. Feature Engineering

```python
tips["is_dinner"] = (tips["time"] == "Dinner").astype(int)
tips["bill_per_person"] = tips["total_bill"] / tips["size"]

X2 = tips[["total_bill", "size", "is_dinner", "bill_per_person"]]
y2 = tips["tip"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model2 = LinearRegression()
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

rmse2 = np.sqrt(mean_squared_error(y2_test, y2_pred))
r2_2 = r2_score(y2_test, y2_pred)

print(f"RMSE (new features) : ${rmse2:.2f}  — was ${rmse:.2f}")
print(f"R²   (new features) : {r2_2:.2f}  — was {r2:.2f}")
```

Explain:

"We did not add new data. We created new columns from what we already had. That is feature engineering — turning domain knowledge into model inputs."

---

### Part 2: Gradient Boosted Trees on Breast Cancer

#### 4. Load the Dataset

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()

print("Target names:", data.target_names)
print("Shape:", data.data.shape)

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df["target"].value_counts()
```

Explain:

"569 real cases, 30 features from tumour biopsies. Binary classification: malignant or benign. The stakes of prediction matter here — a missed cancer is very different from a false alarm."

---

#### 5. Train a Baseline and a GBT Model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Gradient boosting
gbt = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbt.fit(X_train, y_train)
gbt_pred = gbt.predict(X_test)

print(f"Logistic Regression: {accuracy_score(y_test, lr_pred):.2%}")
print(f"Gradient Boosting:   {accuracy_score(y_test, gbt_pred):.2%}")
```

---

#### 6. Read the Classification Report

```python
print(classification_report(y_test, gbt_pred, target_names=data.target_names))
```

Walk through each column:

* **Precision** — of all cases we called malignant, how many were correct?
* **Recall** — of all actual malignant cases, how many did we catch?
* **F1** — the balance between the two
* **Support** — how many real cases of each class in the test set

Ask students: "In a medical context, which metric matters most?" → Recall for malignant.

---

#### 7. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    X, y, cv=5, scoring="accuracy"
)

print(f"CV scores : {scores.round(3)}")
print(f"Mean      : {scores.mean():.3f}")
print(f"Std dev   : {scores.std():.3f}")
```

Explain:

"We trained and tested on five different data splits. Similar scores across all five means the model is stable — it is not just getting lucky on one particular test set."

---

### Student Mini Challenge

Ask students to:

* Add the `smoker`, `day`, or `sex` column to the tips regression model
* Encode them with `pd.get_dummies(tips, columns=["smoker"], drop_first=True)`
* Retrain and compare R² before and after

Bonus:

* Try changing `n_estimators` in the GBT model — does 50 trees perform differently from 200?
* Run `cross_val_score` on the logistic regression baseline and compare the mean to GBT
