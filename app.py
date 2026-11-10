import streamlit as st

st.set_page_config(
    page_title="S5 ML Lab",
    page_icon="💻"
)

st.title("Machine Learning Lab Portal")

st.header("B.Tech S5 - Computer Science")

st.write("Welcome to our ML Lab Website!")

st.subheader("Machine Learning Experiments")

st.write("Experiment 1 - Naive Bayes")
st.write("Experiment 2 - Classification")
st.write("Experiment 3 - Data Preprocessing")
st.write("Experiment 4 - Data Visualization")
st.write("Experiment 5 - Model Evaluation")
st.divider()

st.header("Experiment 1: Naive Bayes Classification")

st.subheader("Python Program")

st.code("""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

categories = [
    'sci.space',
    'rec.sport.hockey',
    'comp.graphics',
    'talk.politics.misc'
]

train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = tfidf.fit_transform(train.data)
X_test = tfidf.transform(test.data)

y_train, y_test = train.target, test.target

print(f"Train : {X_train.shape}\\nTest : {X_test.shape}")

models = [
    ("BernoulliNB", BernoulliNB()),
    ("MultinomialNB", MultinomialNB())
]

for name, model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, target_names=categories))

mnb = MultinomialNB().fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(mnb, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()
""")
st.divider()

st.header("Experiment 2: Polynomial Regression")

st.subheader("Python Program")

st.code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=400,
    n_features=1,
    noise=20,
    random_state=0
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print(f"{'Degree':<10} {'Train MSE':<15} {'Test MSE':<15} {'R2 (Test)'}")
print("-" * 55)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='gray', alpha=0.5, label='Test Data')

x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
colors_list = ['blue', 'green', 'red', 'purple', 'orange']

for deg, color in zip(range(1, 6), colors_list):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])

    pipe.fit(X_train, y_train)

    train_mse = mean_squared_error(y_train, pipe.predict(X_train))
    test_mse = mean_squared_error(y_test, pipe.predict(X_test))
    r2 = r2_score(y_test, pipe.predict(X_test))

    print(f"{deg:<10} {train_mse:<15.2f} {test_mse:<15.2f} {r2:<10.4f}")

    plt.plot(
        x_range, pipe.predict(x_range),
        color=color, label=f'Degree {deg}', linewidth=2
    )

plt.xlabel("Feature (Displacement proxy)")
plt.ylabel("MPG")
plt.title("Polynomial Regression: Degree Comparison")
plt.legend()
plt.tight_layout()
plt.show()
""")
st.divider()

st.header("Experiment 3: Ridge and Lasso Regression")

st.subheader("Python Program")

st.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

alpha_grid = np.logspace(-4, 4, 100)

lr = make_pipeline(
    StandardScaler(), LinearRegression()
).fit(X_train, y_train)

ridge = make_pipeline(
    StandardScaler(), RidgeCV(alphas=alpha_grid, cv=5)
).fit(X_train, y_train)

lasso = make_pipeline(
    StandardScaler(), LassoCV(
        alphas=alpha_grid, cv=5, max_iter=10000
    )
).fit(X_train, y_train)

print(f"{'Model':10s} | {'MSE':>10s} | {'R²':>8s}")
print("-" * 36)

for name, model in [("Linear", lr), ("Ridge", ridge), ("Lasso", lasso)]:
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name:10s} | {mse:10.2f} | {r2:8.4f}")

print("\\n" + "=" * 36)
print(f"Best Ridge alpha: {ridge.named_steps['ridgecv'].alpha_:.4f}")
print(f"Best Lasso alpha: {lasso.named_steps['lassocv'].alpha_:.4f}")
print("=" * 36)

plt.figure(figsize=(10, 5))

plt.plot(lr.named_steps['linearregression'].coef_, 'bo-', label='Linear')
plt.plot(ridge.named_steps['ridgecv'].coef_, 'rs-', label='Ridge')
plt.plot(lasso.named_steps['lassocv'].coef_, 'g^-', label='Lasso')

plt.axhline(0, color='k', linewidth=0.8, linestyle='--')
plt.xlabel("Feature Index")
plt.ylabel("Standardized Coefficient Value")
plt.title("Coefficient Comparison - Ridge vs Lasso vs Linear")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
""")
st.divider()

st.header("Experiment 4: Logistic Regression")

st.subheader("Python Program")

st.code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "MLE": LogisticRegression(C=1e6, max_iter=5000),
    "L2-MAP": LogisticRegression(penalty="l2", C=1.0, max_iter=5000),
    "L1-MAP": LogisticRegression(
        penalty="l1", solver="liblinear", C=1.0, max_iter=5000
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred))

    zeros = (model.coef_ == 0).sum()
    print(f"Zero coefficients: {zeros}\\n")

mle = models["MLE"]
l2 = models["L2-MAP"]
l1 = models["L1-MAP"]

idx = np.arange(X_train.shape[1])

plt.figure(figsize=(12, 4))
plt.bar(idx - 0.25, mle.coef_[0], 0.25, label="MLE")
plt.bar(idx, l2.coef_[0], 0.25, label="L2")
plt.bar(idx + 0.25, l1.coef_[0], 0.25, label="L1")

plt.xlabel("Feature Index")
plt.ylabel("Coefficient")
plt.title("MLE vs L1 vs L2 Coefficients")
plt.legend()
plt.tight_layout()
plt.show()
""")