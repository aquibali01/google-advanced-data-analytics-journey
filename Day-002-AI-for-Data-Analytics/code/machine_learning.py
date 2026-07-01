import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# --- 1. Load Data ---
print("Loading data...")
df = pd.read_csv("synthetic_fraud_dataset.csv")

# --- 2. Feature Engineering & Preprocessing ---
print("Preprocessing features...")

# Drop structural unique IDs that don't help prediction
features_to_drop = ["transaction_id", "user_id"]
X = df.drop(columns=[col for col in features_to_drop if col in df.columns] + ["is_fraud"])
y = df["is_fraud"]

# One-hot encode categorical features (transaction_type, merchant_category, country)
X = pd.get_dummies(X, drop_first=True)

# --- 3. Split into Train/Test Sets ---
# Stratify=y is critical to ensure both sets have the exact same ratio of fraud cases
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. Handle Class Imbalance ---
# Calculate the ratio of negative (legitimate) to positive (fraud) cases
# XGBoost uses this to penalize missing a fraud case much more severely
num_legit = np.sum(y_train == 0)
num_fraud = np.sum(y_train == 1)
scale_weight_value = num_legit / num_fraud
print(f"Class Imbalance Ratio Calculated: {scale_weight_value:.2f}")

# --- 5. Initialize and Train XGBoost Classifier ---
print("Training XGBoost Model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_weight_value,  # Heavy penalty for missing fraud
    eval_metric="aucpr",  # Optimize for Precision-Recall Area Under Curve
    random_state=42,
    use_label_encoder=False,
)

model.fit(X_train, y_train)

# --- 6. Model Evaluation ---
print("\n=== Evaluating Model Performance ===")
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# Display standard classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"]))

# Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"True Negatives (Legit caught): {cm[0,0]}")
# This represents poor customer experience (innocent blocked):
print(f"False Positives (False Alarms): {cm[0,1]}")
# This represents financial loss (stolen money):
print(f"False Negatives (Missed Fraud): {cm[1,0]}")
print(f"True Positives (Fraud caught):  {cm[1,1]}")

# --- 7. Output Feature Importances ---
print("\nTop Predictive Features:")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(5))

# --- 8. Save Model and Metadata for Deployment ---
print("\nSaving model artifact...")
joblib.dump(model, "xgboost_fraud_model.pkl")
# Save columns layout to ensure incoming API calls match exact shape
joblib.dump(X.columns.tolist(), "model_features_layout.pkl")
print("Model saved as 'xgboost_fraud_model.pkl' successfully!")