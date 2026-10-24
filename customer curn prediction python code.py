"""
Customer Churn Prediction - Machine Learning Project
Dataset: customer_churn_dataset_1000_rows.csv
Note: This dataset is synthetic and intended for learning/demo use.
Run:
    pip install pandas scikit-learn matplotlib seaborn joblib
    python customer_churn_prediction.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score
)
import joblib

DATA_PATH = Path(__file__).with_name("customer_churn_dataset_1000_rows.csv")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nChurn counts:")
print(df["Churn"].value_counts())

# Remove ID because it does not describe customer behavior.
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
print("\nClassification report:")
print(classification_report(y_test, pred, target_names=["Retained", "Churned"]))

cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained", "Churned"]).plot()
plt.title("Customer Churn Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig("churn_confusion_matrix.png", dpi=150)
plt.show()

# Save the trained model for later use.
joblib.dump(model, "customer_churn_model.joblib")
print("\nSaved model: customer_churn_model.joblib")
print("Saved chart: churn_confusion_matrix.png")

# Example prediction for one existing customer.
sample_customer = X_test.iloc[[0]]
probability = model.predict_proba(sample_customer)[0, 1]
prediction = "Likely to churn" if probability >= 0.5 else "Likely to stay"
print("\nExample customer prediction:", prediction)
print("Estimated churn probability:", f"{probability:.1%}")
