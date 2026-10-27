"""Customer Churn Prediction & Retention Analysis — Day 36.
Install: python -m pip install pandas numpy matplotlib seaborn scikit-learn
Run: python customer_churn_prediction.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay)

DATA_PATH = None  # Set to a local CSV path if preferred.
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH or DATA_URL)
df.columns = [c.strip() for c in df.columns]
print("Preview:\\n", df.head(), "\\nShape:", df.shape)
print("Missing values:\\n", df.isna().sum(), "\\nDuplicates:", df.duplicated().sum())
if "Churn" not in df:
    raise ValueError("Dataset must contain a Churn column with Yes/No values.")
df = df.drop_duplicates().copy()
df["Churn"] = df["Churn"].astype(str).str.strip()
df = df[df["Churn"].isin(["Yes", "No"])].copy()
df["Churn_Binary"] = df["Churn"].map({"No": 0, "Yes": 1})
if "TotalCharges" in df:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

X = df.drop(columns=[c for c in ["customerID", "Churn", "Churn_Binary"] if c in df])
y = df["Churn_Binary"]
numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = X.select_dtypes(exclude=np.number).columns.tolist()
preprocess = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
out = Path("customer_churn_outputs")
out.mkdir(exist_ok=True)

# EDA plots
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(out / filename, dpi=150)
    plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Churn", order=["No","Yes"])
plt.title("Customer Churn Distribution")
save_plot("01_churn_distribution.png")
if "tenure" in df:
    plt.figure(figsize=(7,4))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=20, multiple="stack")
    plt.title("Tenure by Churn")
    save_plot("02_tenure_by_churn.png")
if "MonthlyCharges" in df:
    plt.figure(figsize=(7,4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", order=["No","Yes"])
    plt.title("Monthly Charges by Churn")
    save_plot("03_monthly_charges_by_churn.png")
if "Contract" in df:
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x="Contract", hue="Churn")
    plt.title("Contract Type vs Churn")
    save_plot("04_contract_vs_churn.png")
if "InternetService" in df:
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x="InternetService", hue="Churn")
    plt.title("Internet Service vs Churn")
    save_plot("05_internet_service_vs_churn.png")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.25, random_state=42, stratify=y)
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42,
                                             class_weight="balanced", n_jobs=-1)
}
models, rows = {}, []
for name, clf in classifiers.items():
    model = Pipeline([("preprocess", preprocess), ("classifier", clf)])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                 "Churn Precision": precision_score(y_test, pred, zero_division=0),
                 "Churn Recall": recall_score(y_test, pred, zero_division=0),
                 "Churn F1": f1_score(y_test, pred, zero_division=0),
                 "ROC-AUC": roc_auc_score(y_test, prob)})
    models[name] = model
    print("\\n", name, "\\n", classification_report(y_test, pred,
          target_names=["No Churn", "Churn"], zero_division=0))
    ConfusionMatrixDisplay.from_predictions(y_test, pred,
        display_labels=["No Churn", "Churn"], values_format="d")
    plt.title("Confusion Matrix — " + name)
    save_plot("confusion_" + name.lower().replace(" ", "_") + ".png")

comparison = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
print("\\nMODEL COMPARISON\\n", comparison.to_string(index=False))
comparison.to_csv(out / "model_comparison.csv", index=False)
df.to_csv(out / "cleaned_churn_dataset.csv", index=False)

plt.figure(figsize=(7,5))
for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=plt.gca())
plt.plot([0,1], [0,1], "--", label="Random baseline")
plt.title("ROC-AUC Model Comparison")
save_plot("roc_auc_comparison.png")

rf = models["Random Forest"]
feature_names = rf.named_steps["preprocess"].get_feature_names_out()
importance = pd.DataFrame({"Feature": feature_names,
    "Importance": rf.named_steps["classifier"].feature_importances_})
importance = importance.sort_values("Importance", ascending=False)
importance.to_csv(out / "feature_importance.csv", index=False)
plt.figure(figsize=(9,5))
sns.barplot(data=importance.head(15), x="Importance", y="Feature")
plt.title("Top Random Forest Feature Importances")
save_plot("feature_importance.png")
print("Top features:\\n", importance.head(15).to_string(index=False))

def predict_churn(customer_record: dict, model=None) -> dict:
    """Return predicted churn class, probability, and illustrative risk band."""
    chosen = model or models["Random Forest"]
    probability = float(chosen.predict_proba(pd.DataFrame([customer_record]))[0,1])
    return {"prediction": "Likely to churn" if probability >= .5 else "Likely to stay",
            "churn_probability": round(probability, 4),
            "risk_band": "High" if probability >= .70 else
                         "Medium" if probability >= .40 else "Low"}

print("Sample prediction:", predict_churn(X_test.iloc[0].to_dict()))
(out / "retention_recommendations_template.txt").write_text(
    "Complete from actual results:\\n1. Best ROC-AUC and churn recall?\\n"
    "2. Which features rank highest?\\n3. Which groups show higher churn?\\n"
    "4. What fair, relevant retention actions could be tested?\\n"
    "5. What are the model limitations?\\n", encoding="utf-8")
print("Outputs saved to", out.resolve())
