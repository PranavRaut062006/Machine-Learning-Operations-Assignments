
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create results directory if it does not exist
os.makedirs("results", exist_ok=True)

# Load the Breast Cancer Wisconsin dataset
dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name="target")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Breast Cancer Classification Model")
print("-" * 40)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of features: {X.shape[1]}")
print(f"Model accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=dataset.target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model and test data for explainability
joblib.dump(model, "results/random_forest_model.pkl")
X_test.to_csv("results/X_test.csv", index=False)
y_test.to_csv("results/y_test.csv", index=False)

# Plot feature importance
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
importances.sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=300)
plt.close()

print("\nModel saved to results/random_forest_model.pkl")
print("Test data saved to results/")
print("Feature importance plot saved to results/feature_importance.png")
