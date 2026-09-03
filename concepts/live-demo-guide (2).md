# Live Demo Guide — Data Science 101

Quick reference for the instructor. Full context in `data-science-101-live-session.md`.

---

## Before the Session

### Install
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Verify everything loads
```python
import seaborn as sns
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report

print(sklearn.__version__)   # needs 1.4+ for root_mean_squared_error
tips = sns.load_dataset("tips")
print(tips.shape)            # (244, 7)
```

> If sklearn < 1.4: use `np.sqrt(mean_squared_error(...))` instead of `root_mean_squared_error`.

### seaborn dataset fallback
seaborn downloads tips from GitHub. If venue WiFi blocks it:
```bash
# Download at home and bring the CSV
curl -O https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
```
```python
tips = pd.read_csv("tips.csv")   # fallback
```

### Font size
Set editor and terminal font to 18pt+ before class.

---

## Demo Flow

| # | What | Time |
|---|---|---|
| 1 | Load tips dataset, explore with `.head()` and a scatter plot | 5 min |
| 2 | Train linear regression, print RMSE and R² | 8 min |
| 3 | Inspect coefficients | 3 min |
| 4 | Feature engineering — add `is_dinner` and `bill_per_person` | 6 min |
| 5 | Retrain, compare metrics | 3 min |
| 6 | Load breast cancer dataset | 4 min |
| 7 | Train logistic regression baseline | 3 min |
| 8 | Train GBT, compare accuracy | 4 min |
| 9 | Read classification report, walk through precision / recall | 5 min |
| 10 | Cross-validation | 4 min |
| **Total** | | **~45 min** |

---

## Step-by-Step Commands

### Step 1 — Load and explore

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

print(tips.shape)
tips.head()
```

```python
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.title("Bill vs Tip")
plt.show()
```

> Ask: "Does this look linear? What do you notice about the spread?"

---

### Step 2 — Train linear regression

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

Expected: `RMSE ~$1.03`, `R² ~0.47`

> Say: "RMSE is in dollars — same unit as the target. R² of 0.47 means we explain 47% of the variance. Better than guessing, but not great."

---

### Step 3 — Inspect coefficients

```python
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
```

> Say: "For every extra dollar on the bill, the model predicts $0.09 more tip. Linear regression is transparent — you see exactly what it learned."

---

### Step 4 — Feature engineering

```python
tips["is_dinner"] = (tips["time"] == "Dinner").astype(int)
tips["bill_per_person"] = tips["total_bill"] / tips["size"]
```

> Say: "We didn't add new data. We created new columns from what we already had. That's feature engineering."

---

### Step 5 — Retrain with new features

```python
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

print(f"RMSE: ${rmse2:.2f}  was ${rmse:.2f}")
print(f"R²  : {r2_2:.2f}  was {r2:.2f}")
```

Expected: modest improvement (~R² 0.52)

> Say: "Small but real. This is what feature engineering looks like in practice — rarely a 2x jump, but meaningful progress."

---

### Step 6 — Load breast cancer dataset

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
print("Target names:", data.target_names)
print("Shape:", data.data.shape)

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
print(df["target"].value_counts())
```

> Say: "569 real cases, 30 measurements from tumour biopsies. Malignant or benign. The stakes make our metric choice matter."

---

### Step 7 — Logistic regression baseline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)

print(f"Logistic Regression: {accuracy_score(y_test, lr.predict(X_test)):.2%}")
```

Expected: ~95–96%

> Ask: "Is 95% good enough for cancer detection? What if those 5% errors are all missed cancers?"

---

### Step 8 — Gradient boosted trees

```python
from sklearn.ensemble import GradientBoostingClassifier

gbt = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbt.fit(X_train, y_train)
gbt_pred = gbt.predict(X_test)

print(f"Gradient Boosting: {accuracy_score(y_test, gbt_pred):.2%}")
```

Expected: ~97%

---

### Step 9 — Classification report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, gbt_pred, target_names=data.target_names))
```

Walk through each column out loud:

- **Precision** — of all cases we flagged malignant, how many were correct?
- **Recall** — of all actual malignant cases, how many did we catch?
- **F1** — the balance between the two
- **Support** — how many real cases of each class in the test set

> Ask: "For a doctor, which metric matters most?" → Recall for malignant.

---

### Step 10 — Cross-validation

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

> Say: "Five different splits, similar scores each time. Low standard deviation means the model isn't getting lucky — it's actually learned the pattern."

---

## Student Mini Challenge

Tell students:

> "Add one more column to the tips regression. Try `smoker`, `day`, or `sex`. You'll need `pd.get_dummies()` to convert the text values to numbers. Does R² improve?"

```python
# Hint for students
tips_encoded = pd.get_dummies(tips, columns=["smoker"], drop_first=True)
print(tips_encoded.columns.tolist())
```

Bonus: "Change `n_estimators` to 50 and 200 in the GBT model. What happens to accuracy?"

---

## If Things Go Wrong

| Problem | Fix |
|---|---|
| `sns.load_dataset("tips")` fails (network) | `tips = pd.read_csv("tips.csv")` with pre-downloaded file |
| `root_mean_squared_error` not found | `np.sqrt(mean_squared_error(y_test, y_pred))` |
| `ConvergenceWarning` on logistic regression | Already handled — `max_iter=10000` is set |
| GBT takes too long | Reduce to `n_estimators=50` |
| R² comes out negative | Check `random_state=42` is set on the split; check X and y are aligned |
| Student gets different numbers | Remind everyone to use `random_state=42` — without it, splits differ |
