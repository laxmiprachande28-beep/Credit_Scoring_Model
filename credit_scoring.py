import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("german_credit_data.csv")


df.columns = [
    "Checking_Account",
    "Duration",
    "Credit_History",
    "Purpose",
    "Credit_Amount",
    "Savings_Account",
    "Employment",
    "Installment_Rate",
    "Personal_Status",
    "Guarantor",
    "Residence_Since",
    "Property",
    "Age",
    "Other_Installment",
    "Housing",
    "Existing_Credits",
    "Job",
    "Dependents",
    "Telephone",
    "Foreign_Worker",
    "Credit_Risk"
]


print("First 5 Rows of Dataset")
print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTarget Variable:")
print(df["Credit_Risk"].value_counts())


X = df.drop("Credit_Risk", axis=1)
y = df["Credit_Risk"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


print("\n==============================")
print("LOGISTIC REGRESSION MODEL")
print("==============================")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

y_prob = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, y_prob)

print("\nModel Evaluation")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC-AUC  :", roc)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\n==============================")
print("RANDOM FOREST MODEL")
print("==============================")

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_prob = rf_model.predict_proba(X_test)[:, 1]
rf_roc = roc_auc_score(y_test, rf_prob)

print("\nModel Evaluation")
print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)
print("ROC-AUC  :", rf_roc)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print("Logistic Regression Accuracy :", round(accuracy * 100, 2), "%")
print("Random Forest Accuracy       :", round(rf_accuracy * 100, 2), "%")
models = ["Logistic Regression", "Random Forest"]
accuracies = [accuracy * 100, rf_accuracy * 100]

plt.figure(figsize=(6, 4))
plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")

for i, value in enumerate(accuracies):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.show()
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(feature_importance)
plt.figure(figsize=(10,6))

plt.bar(feature_importance["Feature"], feature_importance["Importance"])

plt.xticks(rotation=90)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.tight_layout()

plt.show()

print("\n==============================")
print("CREDIT RISK PREDICTION")
print("==============================")

checking_account = int(input("Checking Account (1-4): "))
duration = int(input("Loan Duration (months): "))
credit_history = int(input("Credit History (0-4): "))
purpose = int(input("Purpose (number): "))
credit_amount = int(input("Credit Amount: "))
savings_account = int(input("Savings Account (1-5): "))
employment = int(input("Employment (1-5): "))
installment_rate = int(input("Installment Rate (1-4): "))
personal_status = int(input("Personal Status (1-4): "))
guarantor = int(input("Guarantor (1-3): "))
residence_since = int(input("Residence Since (1-4): "))
property_value = int(input("Property (1-4): "))
age = int(input("Age: "))
other_installment = int(input("Other Installment (1-3): "))
housing = int(input("Housing (1-3): "))
existing_credits = int(input("Existing Credits: "))
job = int(input("Job (1-4): "))
dependents = int(input("Dependents: "))
telephone = int(input("Telephone (1-2): "))
foreign_worker = int(input("Foreign Worker (1-2): "))

new_customer = [[
    checking_account,
    duration,
    credit_history,
    purpose,
    credit_amount,
    savings_account,
    employment,
    installment_rate,
    personal_status,
    guarantor,
    residence_since,
    property_value,
    age,
    other_installment,
    housing,
    existing_credits,
    job,
    dependents,
    telephone,
    foreign_worker
]]

prediction = rf_model.predict(new_customer)

if prediction[0] == 1:
    print("\nPrediction: Good Credit Risk")
else:
    print("\nPrediction: Bad Credit Risk")